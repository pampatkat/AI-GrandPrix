# POINTS ON SPEC-DRIVEN DEVELOPMENT (Educational)

When writing specs like this, include these elements:

1. **Clear Functional Requirements**: What inputs produce what outputs? Be specific (e.g., "2.0 m/s" not "fast").

2. **Non-Functional Requirements**: Frequency, latency, frame choice, speed ranges. Often more important than you think.

3. **Mathematical Foundation**: When transformations are involved (rotations, frame changes), spell out the math. Reduces ambiguity.

4. **Function Signatures**: Exact inputs, outputs, and data types. Implementer shouldn't have to guess.

5. **Edge Cases & Error Handling**: What breaks? How do you handle missing data, conflicts, or unusual states?

6. **Examples & Test Cases**: Walk through concrete scenarios. A good example is worth 1000 words of explanation.

7. **Scope Boundaries**: Explicitly say what's IN and what's OUT. Prevents scope creep.

8. **References to Existing Code**: Link to APIs, constants, and patterns already in the codebase.

9. **Verification/Testing Strategy**: How will you know it works? Specific test cases, not vague statements.