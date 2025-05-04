# Contributing to DeePTB

We heartily welcome contributions to the DeePTB project. This guide provides technical and non-technical guidelines to help you contribute effectively.

## Table of Contents

- [Got a question?](#got-a-question)
- [Project Structure](#project-structure)
- [Submitting an Issue](#submitting-an-issue)
- [Comment Style for Documentation](#comment-style-for-documentation)
- [Code Formatting Style](#code-formatting-style)
- [Adding a Unit Test](#adding-a-unit-test)
- [Running Unit Tests](#running-unit-tests)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [After Your Pull Request is Merged](#after-your-pull-request-is-merged)
- [Commit Message Guidelines](#commit-message-guidelines)

## Got a question?

For questions and discussions about DeePTB, use our [GitHub Discussions](https://github.com/deepmodeling/DeePTB/discussions). If you find a bug or want to propose a new feature, please use the [issue tracker](https://github.com/deepmodeling/DeePTB/issues/new/choose).

## Project Structure

DeePTB is organized into several modules. Here's a brief overview:

- `data`: Data processing module.
- `entrypoints`: Entry points for the command-line interface.
- `negf`: Nonequilibrium Green's Function (NEGF) module.
- `nn`: Neural network model module.
- `nnops`: Neural network operations module.
- `plugins`: Plugins for various functionalities.
- `postprocess`: Post-processing module.
- `tests`: Unit tests for DeePTB.
- `utils`: Utility module with tools and constants.

## Submitting an Issue

Before you submit an issue, please search the issue tracker, and maybe your problem has been discussed and fixed. You can [submit new issues](https://github.com/deepmodeling/DeePTB/issues/new/choose) by filling our issue forms.
To help us reproduce and confirm a bug, please provide a test case and building environment in your issue.


## Comment Style for Documentation

We encourage you to add comments to your code, especially for complex logic or decisions. Use clear, concise language and reference any related issues or discussions.

## Code Formatting Style

Adhere to the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide for Python code. For other languages, follow their respective community standards.

## Adding a Unit Test

When contributing a new feature or fixing a bug, it's important to include tests to ensure the code works as expected and to prevent future regressions.

1. Locate the appropriate test file in the `tests` directory, using `pytest` for Python tests.
2. Write your test case, following the existing structure and style.

## Running Unit Tests

To run all unit tests, use the following command in the project root directory:

```bash
pytest ./dptb/tests
```

To run a specific test, use:

```bash
pytest ./dptb/tests/test_file.py
```

## Submitting a Pull Request

1. [Fork](https://docs.github.com/en/github/getting-started-with-github/fork-a-repo) the [DeePTB repository](https://github.com/deepmodeling/DeePTB). If you already had an existing fork, [sync](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork) the fork to keep your modification up-to-date.
2. Create a new branch for your changes.
     ```shell
     git checkout -b my-fix-branch
     ```
3. Make your changes, including tests and documentation updates.
4. Commit your changes with a [proper commit message](#commit-message-guidelines).
5. Push your branch to your fork on GitHub.
6. Submit a pull request (PR) with `deepmodeling/DeePTB:main` as the base repository. It is required to document your PR following [our guidelines](#commit-message-guidelines).

### After Your Pull Request is Merged

- Delete the remote branch on GitHub either [through the GitHub web UI](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-branches-in-your-repository/deleting-and-restoring-branches-in-a-pull-request#deleting-a-branch-used-for-a-pull-request) or your local shell as follows:

    ```shell
    git push origin --delete my-fix-branch
    ```

- Check out the master branch:

    ```shell
    git checkout develop -f
    ```

- Delete the local branch:

    ```shell
    git branch -D my-fix-branch
    ```

- Update your master with the latest upstream version:

    ```shell
    git pull --ff upstream develop
    ```

## Commit Message Guidelines
A well-formatted commit message leads a more readable history when we look through some changes, and helps us generate change log.

We follow the [Conventional Commits](https://www.conventionalcommits.org) specification for commit messages:

```text
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Example:

```text
Fix(data): handle missing data gracefully

When a required data file is missing, the program now displays a user-friendly error message and exits.

Fixes #123
```

- Header
  - type: The general intention of this commit
    - `Feature`: A new feature
    - `Fix`: A bug fix
    - `Docs`: Only documentation changes
    - `Style`: Changes that do not affect the meaning of the code
    - `Refactor`: A code change that neither fixes a bug nor adds a feature
    - `Perf`: A code change that improves performance
    - `Test`: Adding missing tests or correcting existing tests
    - `Build`: Changes that affect the build system or external dependencies
    - `CI`: Changes to our CI configuration files and scripts
    - `Revert`: Reverting commits
  - scope: optional, could be the module which this commit changes; for example, `orbital`
  - description: A short summary of the code changes: tell others what you did in one sentence.
- Body: optional, providing detailed, additional, or contextual information about the code changes, e.g. the motivation of this commit, referenced materials, the coding implementation, and so on.
- Footer: optional, reference GitHub issues or PRs that this commit closes or is related to. [Use a keyword](https://docs.github.com/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue#linking-a-pull-request-to-an-issue-using-a-keyword) to close an issue, e.g. "Fix #753".

