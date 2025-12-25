# Tasks: Clean Architecture Conversion and Ollama Preparation

**Input**: Design documents from `/specs/002-clean-arch-ollama/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts.md

**Tests**: No test tasks included as not explicitly requested in the feature specification.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/`, `tests/` at repository root
- Paths shown below assume single project - adjust based on plan.md structure

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [X] T001 Create Clean Architecture directory structure in src/ (entities/, use_cases/, interface_adapters/, frameworks_drivers/, shared/)
- [X] T002 [P] Verify Python 3.12+ and existing dependencies (FastAPI, uvicorn, pydantic, requests, pytest)
- [X] T003 [P] Configure linting and formatting tools (if needed)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T004 [P] Create Model entity in src/entities/model.py
- [X] T005 [P] Create Server entity in src/entities/server.py
- [X] T006 [P] Create Agent entity in src/entities/agent.py
- [X] T007 [P] Create Message entity in src/entities/message.py
- [X] T008 [P] Create ModelRepository protocol in src/shared/protocols.py
- [X] T009 [P] Create LLMService protocol in src/shared/protocols.py
- [X] T010 [P] Create AgentManagerInterface protocol in src/shared/protocols.py

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Convert to Clean Architecture (Priority: P1) 🎯 MVP

**Goal**: Convert the codebase to follow Clean Architecture principles with proper layer separation and dependency direction.

**Independent Test**: Can be tested by verifying that code is organized in layers with dependencies pointing inward, and unit tests can run without external dependencies.

### Implementation for User Story 1

- [X] T011 [P] [US1] Create ProcessChatCompletion use case in src/use_cases/process_chat_completion.py
- [X] T012 [P] [US1] Create GetHealth use case in src/use_cases/get_health.py
- [X] T013 [US1] Create ChatController in src/interface_adapters/chat_controller.py
- [X] T014 [US1] Create HealthController in src/interface_adapters/health_controller.py
- [X] T015 [US1] Create LlamaCppLLMService in src/frameworks_drivers/llama_cpp_service.py
- [X] T016 [US1] Create ModelRepository in src/frameworks_drivers/model_repository.py
- [X] T017 [US1] Create AgentManager implementation in src/frameworks_drivers/agent_manager.py
- [X] T018 [US1] Refactor API routes in src/interface_adapters/api.py to use new controllers
- [X] T019 [US1] Update main.py for dependency injection

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Prepare for Ollama Backend (Priority: P2)

**Goal**: Prepare the system to support Ollama as an alternative LLM backend while maintaining existing functionality.

**Independent Test**: Can be tested by configuring the system to use Ollama backend and verifying chat completions work.

### Implementation for User Story 2

- [X] T020 [P] [US2] Create OllamaLLMService in src/frameworks_drivers/ollama_service.py
- [X] T021 [US2] Update configuration to support backend selection in config.json
- [X] T022 [US2] Create LLMServiceFactory in src/frameworks_drivers/llm_service_factory.py
- [X] T023 [US2] Update dependency injection to use factory for LLM service selection

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - Maintain Existing Functionality (Priority: P3)

**Goal**: Ensure all existing proxy features continue working after the conversion with no regression in functionality.

**Independent Test**: Can be tested by running the existing E2E simulation and verifying it passes.

### Implementation for User Story 3

- [X] T024 [US3] Run existing unit tests to ensure no regressions in tests/test_*.py
- [X] T025 [US3] Run main_sim.py E2E simulation and verify successful completion
- [X] T026 [US3] Test agent plugin functionality with slash commands
- [X] T027 [US3] Verify server pool health monitoring works correctly

**Checkpoint**: All user stories should now be independently functional

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T028 [P] Update documentation in README.md and other documentation for Clean Architecture structure
- [ ] T029 Code cleanup and refactoring for consistency
- [ ] T030 Performance optimization across all layers
- [ ] T031 Security hardening for API endpoints
- [ ] T032 Run quickstart.md validation steps

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after User Story 1 completion - Depends on architecture conversion
- **User Story 3 (P3)**: Can start after User Stories 1 and 2 completion - Depends on full implementation

### Within Each User Story

- Models before services
- Services before endpoints
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, User Story 1 can start
- After User Story 1 completes, User Stories 2 and 3 can proceed
- All tasks within a story marked [P] can run in parallel

---

## Parallel Example: User Story 1

```bash
# Launch all entities for Foundational phase together:
Task: "Create Model entity in src/entities/model.py"
Task: "Create Server entity in src/entities/server.py"
Task: "Create Agent entity in src/entities/agent.py"
Task: "Create Message entity in src/entities/message.py"

# Launch all protocols for Foundational phase together:
Task: "Create ModelRepository protocol in src/shared/interfaces.py"
Task: "Create LLMService protocol in src/shared/interfaces.py"
Task: "Create AgentManagerInterface protocol in src/shared/interfaces.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test User Story 1 independently
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (MVP!)
3. Add User Story 2 → Test independently → Deploy/Demo
4. Add User Story 3 → Test independently → Deploy/Demo
5. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1
   - Developer B: User Stories 2 and 3 (after US1)
3. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence