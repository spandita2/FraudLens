"""
feature_engineering.py

Creates features ONLY from columns present in the PaySim schema, and only
from information available BEFORE/AT the moment a transaction is requested
(see leakage discussion in src/data/eda.py).

--------------------------------------------------------------------------
TEMPORAL / TRAIN-TEST CONTAMINATION — WHY THIS FILE IS STRUCTURED THIS WAY
--------------------------------------------------------------------------
An earlier version of this module computed the per-customer historical
features (`orig_txn_count`, `orig_avg_amount`) as an expanding statistic
over the WHOLE dataset (train + test together), sorted by `step`, and only
split into train/test AFTER that. Combined with a random `train_test_split`,
this let a training-set row's historical average silently incorporate a
transaction that later ended up in the test set — i.e. test-period
information influenced a training feature, and the "held-out" test set
was not actually independent of training. It also made the whole
evaluation unrealistic: a production fraud model never gets to see
transactions from the future when it scores a transaction today.

Fixed by two structural changes (both required together):
  1. train_model.py now splits chronologically by `step` BEFORE any
     customer-history feature is computed (see split_chronologically()
     below), so "test" always means "later in time than every training
     row", matching how the model would actually be deployed.
  2. Per-customer historical aggregates are now produced by
     `CustomerHistoryEncoder`, an explicit fit/transform object mirroring
     the same discipline already used for the sklearn ColumnTransformer
     (scaler/one-hot are fit on train only): `.fit_transform(train_df)`
     builds each customer's running (count, amount-sum) state using ONLY
     training-period transactions, in chronological order, with each row's
     feature value computed BEFORE that row updates the state (so a row
     never sees its own transaction, matching the original expanding/
     shift(1) semantics). `.transform(test_df)` then CONTINUES each
     customer's state from wherever fit() left it, walking forward through
     the test period in chronological order. A test-period row can see a
     customer's training-period history and any of that customer's earlier
     test-period rows (both are genuinely in the past relative to that
     row) but can NEVER see a later test-period row or any training row's
     future value — because state only ever advances forward through time,
     never backward. Fold this together with the chronological split and a
     given row's features cannot be built from anything that happens later
     than that row on the calendar, in either the same or the opposite split.

The fallback value used for a customer's first-ever transaction (no prior
history) is now `self.fallback_avg_`, the median of TRAINING amounts only
— the earlier version used the whole dataset's median, which was itself a
(much smaller) leak of test-period statistics into every "new customer"
row.

Row-level features (amount_log, balance deltas, time features, merchant
flag) depend only on that single row's own raw columns, so they carry no
temporal-ordering risk and are computed once, the same way, for both
splits.
--------------------------------------------------------------------------

Explicitly NOT used as model features: newbalanceOrig, newbalanceDest,
isFlaggedFraud, nameOrig, nameDest (raw identifiers) — see eda.py for why.
These exclusions are unchanged from before this refactor.
"""
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Row-level features — pure functions of a single row's own raw columns.
# Safe to compute on train and test independently, before or after any split.
# ---------------------------------------------------------------------------
def add_row_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      hour_of_day, day        — from `step` (1 step = 1 simulated hour)
      amount_log               — log1p(amount), reduces right-skew
      orig_balance_delta       — oldbalanceOrg - amount (pre-transaction-only)
      orig_balance_ratio       — amount / (oldbalanceOrg + 1) ("drain the
                                  account" signature)
      dest_is_merchant         — 1 if nameDest starts with 'M' (PaySim convention)
    """
    df = df.copy()
    df["hour_of_day"] = df["step"] % 24
    df["day"] = df["step"] // 24
    df["amount_log"] = np.log1p(df["amount"])
    df["orig_balance_delta"] = df["oldbalanceOrg"] - df["amount"]
    df["orig_balance_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1.0)
    df["dest_is_merchant"] = df["nameDest"].astype(str).str.startswith("M").astype(int)
    return df


# ---------------------------------------------------------------------------
# Customer-level historical features — stateful, fit on train, extended on test.
# ---------------------------------------------------------------------------
class CustomerHistoryEncoder:
    """
    Produces, for each row:
      orig_txn_count   — number of transactions this customer (nameOrig) made
                          STRICTLY BEFORE this one
      orig_avg_amount  — that customer's average transaction amount over
                          those strictly-prior transactions (fallback_avg_
                          for a customer's very first transaction)

    Usage mirrors an sklearn transformer:
        enc = CustomerHistoryEncoder()
        train_feats = enc.fit_transform(train_df)   # state built from train only
        test_feats  = enc.transform(test_df)         # state continues into test

    `fit_transform` must be called on the training split (it sorts
    internally by `step`); `transform` must only ever be called on data
    that is chronologically at or after everything `fit_transform` saw, or
    the "no future information" guarantee no longer holds.
    """

    def __init__(self):
        self._counts = {}
        self._sums = {}
        self.fallback_avg_ = None
        self._is_fit = False

    def _walk_forward(self, df: pd.DataFrame) -> pd.DataFrame:
        df_sorted = df.sort_values("step", kind="mergesort").copy()
        counts_out = np.empty(len(df_sorted), dtype=np.int64)
        avg_out = np.empty(len(df_sorted), dtype=np.float64)

        for i, (cust, amt) in enumerate(zip(df_sorted["nameOrig"].values, df_sorted["amount"].values)):
            prior_count = self._counts.get(cust, 0)
            prior_sum = self._sums.get(cust, 0.0)
            counts_out[i] = prior_count
            avg_out[i] = (prior_sum / prior_count) if prior_count > 0 else self.fallback_avg_
            self._counts[cust] = prior_count + 1
            self._sums[cust] = prior_sum + amt

        df_sorted["orig_txn_count"] = counts_out
        df_sorted["orig_avg_amount"] = avg_out
        return df_sorted

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fallback_avg_ = float(df["amount"].median())  # TRAIN-only median
        result = self._walk_forward(df)
        self._is_fit = True
        return result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fit:
            raise RuntimeError(
                "CustomerHistoryEncoder must be fit (via fit_transform on the "
                "training split) before calling transform()."
            )
        return self._walk_forward(df)

    def transform_single(self, name_orig: str):
        """Used by the API at inference time for one live transaction — looks
        up a customer's current history WITHOUT mutating state (an
        unconfirmed transaction shouldn't count as history yet)."""
        prior_count = self._counts.get(name_orig, 0)
        prior_sum = self._sums.get(name_orig, 0.0)
        avg = (prior_sum / prior_count) if prior_count > 0 else self.fallback_avg_
        return prior_count, avg


FEATURE_COLUMNS_NUMERIC = [
    "amount_log",
    "oldbalanceOrg",
    "oldbalanceDest",
    "orig_balance_delta",
    "orig_balance_ratio",
    "hour_of_day",
    "day",
    "dest_is_merchant",
    "orig_txn_count",
    "orig_avg_amount",
]
FEATURE_COLUMNS_CATEGORICAL = ["type"]
TARGET_COLUMN = "isFraud"