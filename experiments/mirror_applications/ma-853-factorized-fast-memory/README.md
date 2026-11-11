# MA-853 — factorized key-role × value-role fast memory

H: small key and value role banks compose heldout associative functions with lower state bytes than independent pair matrices. The Mirror coordinate is the pair (key-role id, value-role id).

PA230 motivates matrix-valued fast memory; PA11 requires MAP/Hadamard binding as a control. Implement independent full pair operators, native role factors, Mirror role codes, and MAP/Hadamard binding vectors. Every factor, index, binding and metadata byte is charged.

Synthetic D=32 with four normalized bipolar key roles and four value roles gives 16 logical pair maps. Train-role examples expose each role; checkerboard pairings are held out. Evaluate 64 random query vectors per heldout pair. Development seed 31; fresh 61/79. No optimizer updates or neural-network training.

C: ordinary native role-factor composition is already the direct algebraic control; MAP/Hadamard binding may be smaller and faster. U: learned role extraction, long-sequence interference, concurrent memory writes and trained DeltaNet behavior.
