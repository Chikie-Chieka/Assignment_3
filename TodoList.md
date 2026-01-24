TODO - DEADLINE 31/1:
    CRITICAL:
        - Read the research on Kyber + Ascon128a to see why they chose 128a, as it still exposes weakness against Quantum (halves key size for bruteforce down to 2^64 which is weak). 
        - Also find more research on Ascon hybrids where applicable.
    HIGH PRIORITY:
        - DONE ~~Migrate to C for faster speed~~
        - DONE ~~Add these models:~~
            ~~+ Standalone:~~
                ~~1. BIKE-L1~~
                ~~2. Classic McEliece~~
                ~~3. FrodoKEM~~
                ~~4. HQC~~
                ~~5. Kyber-512~~
                ~~6. X25519~~
            ~~+ Hybrid~~
                ~~1. Classic McEliece + Ascon-128a~~
                ~~2. FrodoKEM + Ascon-128a~~
                ~~3. HQC + Ascon-128a~~
        - Acceptably satisfied via using systemd-run, CPU is mathematically calculated, while RAM usage is virtually insignificant even at 65536 bytes payload per iteration ~~Create environment to limit CPU cores and RAM~~
    MEDIUM PRIORITY:
        - Read the above ~~Add monitor to track CPU % and have better Memory track (for each iterations instead of accrueing peak RAM)~~
    LOW PRIORTITY but must be done before 25/1:
        - Write the first draft for the paper again, following the guide from this prof: https://www.youtube.com/watch?v=UY7sVKJPTMA
        - Get consultation from professors, asking questions on:
            + How to get peer reviews?
            + Should we publish this elsewhere? this is a relatively new topic and might alr be published if we wait 'til July
            + Get them to rate the first drafts and paper multiple times over the week before deadline

Additional Notes for research consideration: 
1. --aad parameter:
For your specific research objectives, modifying the AAD (Additional Authenticated Data) serves three distinct purposes, ranging from simulation realism to revealing subtle performance differences between your baseline and hybrid models.

Here is why you might want to modify it for your specific objectives:

1. Revealing Throughput Differences (Objective 1: Performance)
This is the most technical and impactful reason for your specific comparison between Ascon-128a (Hybrids) and Ascon-80pq (Baseline).

The Mechanism: Ascon-128a and Ascon-80pq have different "rates" (the amount of data processed per permutation round).
Ascon-128a (used in your Hybrids) processes 16 bytes per round.
Ascon-80pq (your Baseline) processes 8 bytes per round.
The Impact: If you set a large AAD (e.g., simulating a large certificate header), Ascon-128a will process it twice as fast (in terms of rounds) as Ascon-80pq.
Research Value: Modifying AAD allows you to see if the "Security Tax" of the hybrids (the KEM overhead) is partially offset by the faster symmetric processing of Ascon-128a compared to the slower Ascon-80pq baseline during the data phase.
2. Simulating "Resource-Limited Environments" (Objective 1 & 2)
Your research targets IoT and constrained devices. In these environments, data is rarely sent as raw payload; it is wrapped in protocols.

Realism: Protocols like LoRaWAN, Zigbee, or MQTT-SN have unencrypted headers (device IDs, frame counters, control flags) that must be authenticated to prevent replay attacks or routing spoofing.
Configuration: Setting AAD to a realistic size (e.g., 13–20 bytes) makes your latency and memory benchmarks represent a "real-world IoT packet" rather than a theoretical algorithm test.
Security Tax: By including AAD, you measure the total system cost. If the KEM takes 10ms, but processing the packet header (AAD) + payload takes 5ms, the "Tax" is relative to that 15ms total.
3. Verification of Cryptographic Binding (Objective 3: Entropy/Robustness)
While AAD is not encrypted, it influences the ciphertext generation.

Avalanche Effect: Modifying the AAD changes the internal state of the Ascon sponge before encryption begins. Even if the Payload and Key are identical, changing 1 bit of AAD should result in a completely different Ciphertext (and thus different Entropy/SCC stats).
Robustness Check: Ensuring your implementation correctly handles AAD confirms that the "Hybrid" wrapper isn't just gluing components together but is correctly binding the session context (the AAD) to the encrypted data.
Summary Recommendation
For your experiments:

To Compare KEMs (Kyber vs. BIKE vs. X25519): Keep AAD constant (e.g., 0 or 16 bytes) across all runs. Varying it would introduce noise that obscures the KEM differences.
To Compare Baseline vs. Hybrid: Be aware that Ascon-80pq is slower at processing AAD. A non-zero AAD is fairer to the Hybrids because it highlights the efficiency of Ascon-128a.
