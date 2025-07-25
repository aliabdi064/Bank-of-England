# Bank of England Project

## Data Schema

This project uses a dataset with the following columns:

*   **Transaction unique identifier**: A generated ID for each sale.
*   **Price**: The sale price stated on the transfer deed.
*   **Date of Transfer**: When the sale was completed, as stated on the deed.
*   **Property Type**: 
    *   D = Detached
    *   S = Semi-Detached
    *   T = Terraced
    *   F = Flat/Maisonette
    *   O = Other
*   **Old/New**: 
    *   Y = newly built
    *   N = established dwelling
*   **Duration**: Tenure:
    *   F = Freehold
    *   L = Leasehold (only leases over seven years included)
*   **PAON/SAON/Street/Locality/Town/County**: Full address breakdown including postcode.
*   **PPD Category Type**: Categorises the transaction as:
    *   A = Standard Price Paid (single residential sold for value)
    *   B = Additional Price Paid Data (e.g. repossessions, buy-to-lets, transfers to non-private individuals)
*   **Record Status (monthly files only)**: Indicates updates to prior records:
    *   A = addition
    *   C = change
    *   D = deletion
