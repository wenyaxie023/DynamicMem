RQ1. State Reconstruction

RQ1. Can current memory systems reconstruct heterogeneous user states from long-horizon user trajectories?
H1.1 Current memory systems will achieve substantially higher performance on attributes than on habits and preferences.
H1.2 Fine-grained memory extraction improves recall of explicit attribute evidence, but is insufficient for recovering stable habits that require higher-level abstraction.
H1.3 Retrieval-based memory is especially vulnerable on preferences, since the relevant evidence is typically implicit and temporally dispersed rather than directly retrievable through lexical overlap.

RQ2. State Updating
RQ2. Can current memory systems maintain temporally correct and up-to-date user states under long-horizon state evolution?
H2.1 Stale-state dominance
H2.2 Duplicate-without-merge
H2.3 Reversible habits are especially hard

RQ3. Personalization Utility
RQ3. Can current memory systems support personalized service from long-horizon user trajectories?
H3.1 Different memory systems support personalized service differently.
H3.2 Performance differences across memory systems are larger on changed states than on unchanged states.
H3.3 Some memory systems can recover user state but still fail in personalized service.

## Article-Ready Baseline Table

| Baseline | Memory Representation | Update Strategy | Retrieval / Use | Role in This Study |
| --- | --- | --- | --- | --- |
| `RAG` | Raw log chunks | Append-only | Dense retrieval over stored history | Retrieval-based anchor without explicit memory structuring |
| `HippoRAG2` | Graph-structured fact memory | Incremental graph construction | Associative graph retrieval | Tests whether explicit relational structure helps long-horizon state reconstruction |
| `A-MEM` | Linked memory notes | Memory revision with note linking and neighbor updates | Retrieval over connected memory notes | Tests whether explicit memory revision improves state updating |
| `SimpleMem` | Compressed semantic memory entries | Online consolidation and synthesis | Hybrid planned retrieval | Tests whether semantic compression improves storage efficiency while preserving useful user state |
| `MemoryOS` | Hierarchical multi-timescale memory | Cross-level consolidation from short-term to longer-term memory | Multi-store retrieval across memory levels | Tests whether hierarchical memory organization better supports dynamic personalization over time |

## Task C v2 Design Note

Status:
- canonical design direction for the next Task C update
- not yet implemented in prompt / pack / eval code
- purpose: define the target before modifying Task C synthesis prompts and validation prompts

### Motivation

Current Task C is still too close to apply QA. In practice, some items can be answered by using only one salient field from a state, which weakens `RQ3` as a personalization-utility benchmark.

For the next v2 revision, Task C should move from:
- `apply QA over a remembered state`

to:
- `proactive personalized service completion with structured outputs`

The core requirement is:
- an item should only receive full credit when the model applies all required non-derived state fields correctly in a downstream service task

### Paper-Facing Taxonomy

Task C v2 should organize proactive personalized service into three service-interface families:

1. `User Communication`
   - generate a user-facing proactive communication action
   - typical output: reminder / notification / follow-up action object

2. `Preference-Conditioned Filtering Parameter Completion`
   - generate the value of structured filtering parameters for a downstream retrieval / screening / recommendation step
   - typical output: filter values, screening constraints, preference-conditioned parameter fields
   - this task does not ask for a final recommendation; it asks the assistant to fill the filtering parameters that will later guide a downstream information system

3. `Action Configuration`
   - generate structured configuration parameters for a downstream tool / workflow / execution system
   - typical output: tool arguments, enabled resources, sync settings, execution configuration fields

This taxonomy is meant to separate where remembered user state is operationalized:
- user-facing communication
- information-facing request construction
- tool-facing action configuration

It is not meant to claim that these are the only possible assistant behaviors. It is a controlled decomposition for benchmark design.

Runtime compatibility note:
- the stable internal Task C v2 family ids remain:
  - `user_communication`
  - `information_request_construction`
  - `action_configuration`
- the internal id `information_request_construction` is retained for compatibility, but its paper-facing meaning is now `Preference-Conditioned Filtering Parameter Completion`

### Canonical Mapping for Benchmark Construction

For the next Task C revision, item synthesis should use a canonical mapping:

- `habits -> User Communication`
- `preferences -> Preference-Conditioned Filtering Parameter Completion`
- `attributes -> Action Configuration`

This mapping is a benchmark-construction choice, not a claim that each state type can only support one service family. The reason to keep the mapping fixed is to avoid mixing task form and state type in the same headline metric.

### Prediction Shape

Task C v2 should use structured service-completion outputs rather than multiple choice.

Common idea:
- each item provides:
  - a bounded service scenario
  - a concrete service task
  - an output JSON schema
- the model must fill the required fields in that schema

Examples:

- `User Communication`:
  - `{"should_send": true|false, "send_time": "...", "message": "..."}`

- `Preference-Conditioned Filtering Parameter Completion`:
  - `{"filtering_params": {"preference_statement": "..."}}`

- `Action Configuration`:
  - `{"tool_name": "...", "arguments": {...}}`

### Item Validity Rule

Task C v2 should not score `field coverage` as a separate runtime metric. Instead, coverage should be enforced at build time.

Each synthesized item should satisfy:

- `structured-completion`: output is a bounded JSON object, not a free-form answer only
- `full-field dependency`: all required non-derived source fields are needed for the gold output
- `counterfactual necessity`: if one required source field changes, at least one required output slot should change
- `unique gold`: under the given context and schema, the gold output should not have multiple equally valid variants

### Scoring Direction

Primary score:
- `required structured slot accuracy`

This means:
- the main Task C score should be computed over required output slots in the service JSON
- not over A/B/C/D choice accuracy
- not over a whole-answer holistic judge

Auxiliary score:
- optional quality checks for free-text fields such as `message`
- evidence correctness remains auxiliary

### Canonical Examples

`habits -> User Communication`
- use structured habit fields such as `schedule`, `timing`, `location`, `priority`
- output a reminder / notification action object

`preferences -> Preference-Conditioned Filtering Parameter Completion`
- use the preference statement as the source for downstream filtering parameters rather than as a final user-facing answer
- output a structured filtering-parameter object for downstream retrieval / screening / recommendation

`attributes -> Action Configuration`
- use user resources / accounts / tools / assets / roles as service-configuration parameters
- output a tool / workflow configuration object that determines how a downstream service should run

### Why This Update Is Needed Next

The next Task C implementation step will require changes to:

- Task C synthesis prompts
- Task C validation prompts
- Task C output schema
- Task C scoring-point generation

This design note is the source of truth for that upcoming prompt-and-contract refactor.
