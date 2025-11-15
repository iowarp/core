# Contributing to IOWarp Core

## Installation

IOWarp Core uses **DevContainers** to provide a consistent development environment. This eliminates dependency issues by packaging all build tools and compilers into a Docker container.

### Prerequisites

- Docker Desktop (Windows/macOS) or Docker Engine (Linux)
- Visual Studio Code with the **Dev Containers** extension

### Setup Steps

1. **Install Docker**
   - **Windows/macOS**: Install [Docker Desktop](https://docs.docker.com/desktop/)
   - **Linux**: Install [Docker Engine](https://docs.docker.com/engine/install/) and add your user to the docker group:
     ```bash
     sudo usermod -aG docker $USER
     # Log out and back in
     ```

2. **Install VSCode**
   - Install [Visual Studio Code](https://code.visualstudio.com/)
   - Install the **Dev Containers** extension from the marketplace

3. **Clone and Open**
   ```bash
   git clone <repository-url>
   cd iowarp
   code .
   ```

4. **Reopen in Container**
   - When prompted, click **"Reopen in Container"**
   - Or press **F1** → **"Dev Containers: Reopen in Container"**
   - First build takes 5-10 minutes; subsequent opens are instant

5. **Verify**
   ```bash
   gcc --version    # Should show GCC 13.x
   cmake --version  # Should show CMake 3.28+
   whoami          # Should show "iowarp"
   ```

### DevContainer Features

- **Docker-in-Docker**: Build and test containers from within the dev environment
- **SSH key mounting**: Your `~/.ssh` keys are automatically mounted for git operations
- **Pre-installed tools**: GCC 13, Clang 18, CMake, MPI, ZeroMQ, Boost, Catch2
- **VSCode extensions**: clangd, CMake Tools, C++ debugging, Docker integration

### SSH Key Setup

Ensure SSH keys are configured before starting the container:

```bash
# Generate if needed
ssh-keygen -t ed25519 -C "your_email@example.com"

# Add to GitHub/GitLab
cat ~/.ssh/id_ed25519.pub
```

## Repository Structure

Four main development directories (all prefixed with `context-`):

```
iowarp/
├── context-runtime/              ⭐ Core runtime and ChiMod system
├── context-transport-primitives/ ⭐ IPC, shared memory, networking
├── context-transfer-engine/      ⭐ Data staging and I/O adapters
└── context-assimilation-engine/  ⭐ Data transformation pipelines
```

### context-runtime/

**Core Chimaera runtime system** - orchestrates distributed task execution

- `modules/` - ChiMod implementations (admin, bdev)
- `src/` - Runtime core (scheduler, workers, tasks)
- `test/` - Unit and distributed tests
- `cmake/` - Build configuration

Work here for: New ChiMods, scheduling logic, runtime features, distributed tests

### context-transport-primitives/

**Low-level IPC and data structures** - efficient inter-process communication

- `include/hermes_shm/` - Shared memory structures (queues, atomics, allocators)
- `src/` - IPC primitives and network transport
- `docs/` - **MODULE_DEVELOPMENT_GUIDE.md** (essential for ChiMod developers)
- `test/` - Unit tests for data structures

Work here for: Shared memory structures, IPC optimization, network protocols

### context-transfer-engine/

**CTE (Context Transfer Engine)** - data staging and movement between storage tiers

- `core/` - CTE implementation and API
- `adapters/` - I/O backends (POSIX, MPI-IO, STDIO)
- `docs/cte/` - **cte.md** (complete CTE API documentation)
- `test/` - Unit tests and benchmarks

Work here for: I/O adapters, async I/O, data transfer optimization, CTE API extensions

### context-assimilation-engine/

**CAE (Context Assimilation Engine)** - data processing and transformation pipelines

- `core/` - CAE implementation
- `adapters/` - Data transformation adapters
- `test/` - Unit tests

Work here for: Data pipelines, transformation adapters, CTE integration

### Using Documentation with AI

Each `context-*` directory has a `docs/` subdirectory with development guides:

- **context-transport-primitives/docs/MODULE_DEVELOPMENT_GUIDE.md** - ChiMod development
- **context-transfer-engine/docs/cte/cte.md** - CTE API reference
- **CLAUDE.md** (root) - Project-wide development guidelines

**When using AI assistants:** Reference these docs for context. Example: *"Using the patterns in MODULE_DEVELOPMENT_GUIDE.md, help me implement a ChiMod for..."*

## Code Style

IOWarp Core follows **Google C++ Style Guide** with automated formatting.

### clang-format

Configuration in `.clang-format`:
```yaml
BasedOnStyle: Google
UseTab: Never
ColumnLimit: 80
```

**Format before committing:**
```bash
# Single file
clang-format -i src/my_file.cc

# All C++ files
find . -name "*.cc" -o -name "*.h" | xargs clang-format -i
```

**In VSCode:** Right-click → Format Document (Shift+Alt+F)

### CPPLINT

Configuration in `CPPLINT.cfg`. Run linting:
```bash
cpplint src/my_file.cc
```

### VSCode Integration

The DevContainer configures **clang-tidy** for real-time linting:
- `clang-analyzer-*` - Static analysis
- `performance-*` - Performance checks
- `modernize-*` - Modern C++ practices

Linting runs automatically on save (red squiggles show issues).

### Key Style Rules

From `CLAUDE.md`:

1. **Store singleton pointers:**
   ```cpp
   auto *manager = hshm::Singleton<Manager>::GetInstance();
   manager->DoSomething();
   ```

2. **Document all functions (Doxygen):**
   ```cpp
   /**
    * Description of function
    * @param param1 Description
    * @return Description
    */
   ```

3. **Use named constants:**
   ```cpp
   constexpr QueueId kHighPriorityQueue = 0;
   ```

4. **Include units in timing:**
   ```cpp
   HILOG(kInfo, "Operation took {} ms", elapsed);
   ```

5. **No stub code** - Always implement real, working code

## GitHub Contributors (Internal)

For Gnosis engineers and students with repository access:

### Workflow

1. **Create branch from main:**
   ```bash
   git checkout -b feature/my-feature
   # Or: fix/, docs/, refactor/, test/
   ```

2. **Make changes and commit:**
   ```bash
   vim src/my_file.cc
   clang-format -i src/my_file.cc

   cmake --preset=debug
   cmake --build --preset=debug
   cd build && ctest

   git add src/my_file.cc
   git commit -m "Add feature X

   - Implement algorithm Y
   - Add unit tests

   Closes #123"
   ```

3. **Push and create PR:**
   ```bash
   git push -u origin feature/my-feature
   ```
   Then create Pull Request on GitHub.

4. **Address review feedback** and push updates

5. **After merge:**
   ```bash
   git checkout main
   git pull origin main
   ```

### Branch Protection

- Direct pushes to `main` are forbidden
- All changes require Pull Request with approval
- CI/CD checks must pass

## External Collaborators

For external contributors without repository access:

### Fork Workflow

1. **Fork the repository** on GitHub (click Fork button)

2. **Clone your fork:**
   ```bash
   git clone git@github.com:YOUR_USERNAME/iowarp.git
   cd iowarp
   git remote add upstream git@github.com:ORIGINAL_ORG/iowarp.git
   ```

3. **Sync with upstream before starting work:**
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   git push origin main
   ```

4. **Create branch and make changes:**
   ```bash
   git checkout -b feature/my-contribution

   # Make changes, format, build, test
   vim src/my_file.cc
   clang-format -i src/my_file.cc
   cmake --preset=debug && cmake --build --preset=debug
   cd build && ctest

   git add src/my_file.cc
   git commit -m "Add feature: improve XYZ"
   ```

5. **Push to your fork:**
   ```bash
   git push origin feature/my-contribution
   ```

6. **Create PR** from your fork to original repo on GitHub

7. **After merge, sync your fork:**
   ```bash
   git checkout main
   git fetch upstream
   git merge upstream/main
   git push origin main
   ```

### Guidelines

- Read `CLAUDE.md` before contributing
- Follow code style (use clang-format)
- Write tests for new features
- Document your code
- Keep PRs focused (one feature/fix per PR)

---

**Questions?** Open a GitHub issue or discussion.

**Welcome to IOWarp Core!**
