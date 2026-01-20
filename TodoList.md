TODO - DEADLINE 31/1:
    CRITICAL:
        - Read the research on Kyber + Ascon128a to see why they chose 128a, as it still exposes weakness against Quantum (halves key size for bruteforce down to 2^64 which is weak). 
        - Also find more research on Ascon hybrids where applicable.
    HIGH PRIORITY:
        - Migrate to C for faster speed
        - Add these models:
            + Standalone:
                1. BIKE
                2. Classic McEliece
                3. FrodoKEM
                4. HQC
            + Hybrid
                1. Classic McEliece + Ascon-128a
                2. FrodoKEM + Ascon-128a
                3. HQC + Ascon-128a
        - Create environment to limit CPU cores and RAM
    MEDIUM PRIORITY:
        - Add monitor to track CPU % and have better Memory track (for each iterations instead of accrueing peak RAM)
    LOW PRIORTITY but must be done before 25/1:
        - Write the first draft for the paper again, following the guide from this prof: https://www.youtube.com/watch?v=UY7sVKJPTMA
        - Get consultation from professors, asking questions on:
            + How to get peer reviews?
            + Should we publish this elsewhere? this is a relatively new topic and might alr be published if we wait 'til July
            + Get them to rate the first drafts and paper multiple times over the week before deadline


