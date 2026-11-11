# MA-818 status

- Status: FAIL for frozen 25% full-byte gate and Mirror-specific attribution.
- Branch: `research/ma-818-new-how-code-20261111`
- Base: `92c47a48`
- Development: synthetic seed 81801; six content vectors; one new operator fitted from 32 examples.
- Fresh/audit: sealed after native coordinate alias and storage threshold miss.
- Verification: focused test passed; serialized reconstruction replay exact.

## H / T / D / C / U

- **H:** A new how code adds a mechanism cheaper than explicit operators and native coordinates.
- **T:** Frozen rank-2 operator basis; compare explicit matrix, native code and Mirror code.
- **D:** **FAIL.** Mirror is 1,603 B versus 2,088 B explicit (23.2% saving, below 25%) and exactly aliases native code, which is 1,611 B.
- **C:** Standard operator-coordinate fitting explains all gains; serialization labels account for 8 B.
- **U:** Multiple mechanisms, learned bases, natural tasks and fresh seeds.
