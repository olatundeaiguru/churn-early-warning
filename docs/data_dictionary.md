# Data Dictionary: `cell2celltrain.csv`

58 columns. One row = one customer at the moment their data was recorded (the **snapshot**).

## How sure are we?

The full original documentation (Duke University / Teradata Center, "Cell2Cell: The Churn Game") was not available in a readable form. Each meaning is marked:

| Mark | Meaning |
|---|---|
| ✅ | Confirmed by a documentation extract or a published paper |
| 🔸 | Inferred from the column name and the values in the data. **Not confirmed** |

**What the documentation does confirm:**
- The target is churn 31 to 60 days after the snapshot.
- Variables described as "Mean" are **averages over the 4 months before the snapshot**. This is why many call counts have decimals (e.g. 0.7 dropped calls).
- Revenue is mean monthly revenue. Minutes of use is a mean. Equipment days is the number of days of the current equipment.

We do not know exactly which columns are 4-month averages. The usage and call columns (sections 2 and 3) most likely are.

**"Notes" column:** findings from `notebooks/01_data_inspection.ipynb` (all data) and `notebooks/02_eda.ipynb` (training set).

---

## 1. ID and target

| Column | Meaning | Type | Sure? | Notes |
|---|---|---|---|---|
| `CustomerID` | Unique customer number | Number | 🔸 | Unique per row. **Never a model input** |
| `Churn` | Did the customer leave 31 to 60 days after the snapshot? | Yes / No | ✅ | **The target.** 28.8% Yes (decision A1) |

## 2. Billing and usage

| Column | Meaning | Type | Sure? | Notes |
|---|---|---|---|---|
| `MonthlyRevenue` | Average monthly bill (revenue from the customer) | Number ($) | ✅ | Missing for 156 customers. A few negatives (refunds or errors?) |
| `MonthlyMinutes` | Average monthly minutes of phone use | Number | ✅ | Missing for the same 156 customers |
| `TotalRecurringCharge` | Average fixed monthly plan charge | Number ($) | 🔸 | Missing for the same 156. A few negatives |
| `DirectorAssistedCalls` | Average calls to directory assistance (operator help) | Number | 🔸 | Missing for the same 156 |
| `OverageMinutes` | Average minutes used **above** the plan allowance | Number | 🔸 | Missing for the same 156. Very skewed: most near 0, a few huge |
| `RoamingCalls` | Average calls made while roaming (outside home network) | Number | 🔸 | Missing for the same 156 |
| `PercChangeMinutes` | % change in minutes of use (recent vs earlier months) | Number (%) | 🔸 | Missing for 367. **Missing = strong churn warning** (57% churn in train) |
| `PercChangeRevenues` | % change in revenue (recent vs earlier months) | Number (%) | 🔸 | Missing for the same 367. Can be negative (that is fine) |

## 3. Calls and call quality

All look like **monthly averages** (they have decimals).

| Column | Meaning | Type | Sure? | Notes |
|---|---|---|---|---|
| `DroppedCalls` | Calls cut off during the call | Number | 🔸 | Network quality |
| `BlockedCalls` | Calls that could not connect | Number | 🔸 | Network quality |
| `UnansweredCalls` | Calls made that nobody answered | Number | 🔸 | |
| `CustomerCareCalls` | Calls to customer care | Number | 🔸 | Used in our rule baseline (decision A5) |
| `ThreewayCalls` | Three-way (conference) calls | Number | 🔸 | |
| `ReceivedCalls` | Calls received | Number | 🔸 | ⚠️ The original variable may be **minutes** of received calls, not a count. Treat with care |
| `OutboundCalls` | Outgoing calls | Number | 🔸 | |
| `InboundCalls` | Incoming calls | Number | 🔸 | |
| `PeakCallsInOut` | Calls (in and out) at peak times | Number | 🔸 | |
| `OffPeakCallsInOut` | Calls (in and out) at off-peak times | Number | 🔸 | |
| `DroppedBlockedCalls` | Dropped + blocked calls combined | Number | 🔸 | Probably the sum of `DroppedCalls` and `BlockedCalls`. Check in EDA Part C |
| `CallForwardingCalls` | Calls forwarded to another number | Number | 🔸 | Almost always 0 |
| `CallWaitingCalls` | Calls using call waiting | Number | 🔸 | |

## 4. Account and phone (handset)

| Column | Meaning | Type | Sure? | Notes |
|---|---|---|---|---|
| `MonthsInService` | Months the customer has been with the company | Number | 🔸 | Range 6 to 61. **No customers under 6 months** |
| `UniqueSubs` | Number of subscribers (phone lines) on the account | Number | 🔸 | Most have 1 or 2. Top 1% have more than 5 (churn 42%) |
| `ActiveSubs` | Number of those subscribers still active | Number | 🔸 | Never more than `UniqueSubs` |
| `ServiceArea` | Service area code, e.g. `NYCBRO917` (city + area + phone area code) | Text | 🔸 | 747 different areas. Missing for 24 |
| `Handsets` | Number of phones the customer has had | Number | 🔸 | |
| `HandsetModels` | Number of different phone models the customer has had | Number | 🔸 | |
| `CurrentEquipmentDays` | Days the customer has had their current phone (phone age) | Number | ✅ | A few negatives (impossible) |
| `HandsetRefurbished` | Is the current phone refurbished (second-hand, repaired)? | Yes / No | 🔸 | |
| `HandsetWebCapable` | Can the current phone use the internet? | Yes / No | 🔸 | |
| `HandsetPrice` | Price band of the current phone | Text | 🔸 | Values like 10, 30, 150 (probably $). **57% "Unknown"** |

## 5. Relationship with the company

| Column | Meaning | Type | Sure? | Notes |
|---|---|---|---|---|
| `RetentionCalls` | Number of calls the customer made to the retention team | Number | 🔸 | ⚠️ **Possible leakage** (decision A2). Strong churn link |
| `RetentionOffersAccepted` | Number of retention offers the customer accepted | Number | 🔸 | ⚠️ Possible leakage (A2) |
| `MadeCallToRetentionTeam` | Has the customer ever called the retention team? | Yes / No | 🔸 | ⚠️ Possible leakage (A2). Same customers as `RetentionCalls` > 0 |
| `ReferralsMadeBySubscriber` | Number of new customers this customer referred | Number | 🔸 | |
| `NewCellphoneUser` | Is the customer new to mobile phones? | Yes / No | 🔸 | Overlaps with the next column |
| `NotNewCellphoneUser` | Has the customer used mobile phones before? | Yes / No | 🔸 | Both often "No", so together: new / not new / unknown |
| `CellphoneUserStatus` | **Created by cleaning** (A7): the two columns above combined | New / NotNew / Unknown | ✅ | Replaces `NewCellphoneUser` and `NotNewCellphoneUser` after cleaning |
| `CreditRating` | Customer's credit rating | Text | 🔸 | Ordered: 1-Highest to 7-Lowest |
| `AdjustmentsToCreditRating` | Number of changes to the credit rating | Number | 🔸 | |

## 6. Household and personal (demographics)

⚠️ **Personal data.** The columns marked **A4** are left out of the first model and used for the fairness check (decision A4).

| Column | Meaning | Type | Sure? | Notes |
|---|---|---|---|---|
| `AgeHH1` | Age of household member 1 (probably the account holder) | Number | 🔸 | **A4.** 0 = unknown (27% of train) |
| `AgeHH2` | Age of household member 2 | Number | 🔸 | **A4.** 0 = unknown or no second person |
| `ChildrenInHH` | Are there children in the household? | Yes / No | 🔸 | **A4** |
| `IncomeGroup` | Household income band (1 to 9) | Number | 🔸 | **A4.** 0 = unknown, mostly the same customers as age 0 |
| `MaritalStatus` | Is the customer married? | Yes / No / Unknown | 🔸 | **A4.** 39% Unknown |
| `Occupation` | Type of job | Text | 🔸 | **A4.** 74% "Other" |
| `Homeownership` | Is home ownership known? | Known / Unknown | 🔸 | Only says known or unknown, not own or rent |
| `PrizmCode` | Neighbourhood type (Suburban, Town, Rural, Other) | Text | 🔸 | Kept in, but gets a fairness check: may hide income or ethnicity |
| `TruckOwner` | Owns a truck? | Yes / No | 🔸 | Lifestyle data |
| `RVOwner` | Owns a recreational vehicle (camper)? | Yes / No | 🔸 | Lifestyle data |
| `OwnsMotorcycle` | Owns a motorcycle? | Yes / No | 🔸 | Lifestyle data |
| `OwnsComputer` | Owns a computer? | Yes / No | 🔸 | Lifestyle data |
| `HasCreditCard` | Has a credit card? | Yes / No | 🔸 | Lifestyle data |
| `NonUSTravel` | Has travelled outside the US? | Yes / No | 🔸 | Lifestyle data |
| `BuysViaMailOrder` | Buys products by mail order? | Yes / No | 🔸 | Lifestyle data |
| `RespondsToMailOffers` | Responds to offers sent by mail? | Yes / No | 🔸 | Lifestyle data |
| `OptOutMailings` | Has asked not to receive mail offers? | Yes / No | 🔸 | Lifestyle data |

---

## Open questions

1. **Lifestyle columns** (truck, RV, motorcycle, computer, credit card, travel, mail order): these are also personal data about the customer, but they are not on the A4 list. Do we treat them like A4 (leave out first) or keep them? To decide in Stage 5.
2. **`ReceivedCalls`:** count of calls or minutes? Affects how we describe it to the business, not whether we can use it.
3. **Retention columns:** were they recorded before the snapshot? Decision A2, Stage 5.

## Sources

- Third-party copies of the Cell2Cell documentation (Scribd): churn window, "Mean" = 4-month average, revenue and minutes of use descriptions.
- Published churn papers using Cell2Cell: equipment days = days of current equipment.
- All 🔸 meanings: our own reading of column names and values.
