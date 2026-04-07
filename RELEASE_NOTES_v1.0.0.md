# BLADE v1.0.0 Release Notes

This release summarizes all changes since **v0.9.0**.

> Note: the local repository snapshot does not currently include a `v0.9.0`
> git tag. The change range used here is based on commits after
> `7d8bf1a` (`New version (#97)`), which appears to be the v0.9.0 release
> commit in history.

## Highlights

- **Benchmark module refactor and namespace cleanup**
  - Migrated public benchmark modules from `iohblade/problems` to
    `iohblade/benchmarks`, including BBOB, AutoML, KernelTuner,
    and Photonics paths.
  - Updated examples and tests to the new import locations.
  - Related pull request: **#98**.

- **AutoML integration refreshed**
  - Updated the LLaMEA/AutoML integration to track the latest BLADE API.
  - Related pull request: **#81**.

- **Significant benchmark documentation expansion**
  - Added and expanded benchmark-specific documentation and READMEs for
    BBOB, combinatorics, geometry, logistics, packing, photonics,
    and KernelTuner.
  - Improved benchmark coverage and navigation in docs index pages.
  - Related pull requests: **#100**, **#101**, **#102**, **#104**.

- **Documentation publishing improvements**
  - Updated docs deployment workflow to publish on releases.
  - Default docs versioning now resolves to the latest tag.
  - Related pull request: **#103**.

## Pull Requests included

- #81 — Integrate LLaMEA / AutoML updates.
- #98 — Benchmark refactor merge (`iohblade/problems` → `iohblade/benchmarks`).
- #100 — AutoML benchmark documentation details.
- #101 — KernelTuner category/documentation update.
- #102 — Feature coverage and docs navigation improvements.
- #103 — Docs deployment on release + latest-tag default version.
- #104 — Docs index benchmark coverage + version-selector navigation fix.

## Upgrade notes

- **Potentially breaking import-path change**:
  consumers importing from `iohblade.problems` should migrate to
  `iohblade.benchmarks` namespaces.
- If you host documentation, align your deployment with the updated
  release-triggered workflow introduced in #103.

## Contributor-facing changelog summary

Between 2026-02-16 and 2026-03-07, this release delivered:

- 17 commits after the prior version commit (`#97`),
- one large benchmark refactor merge,
- one AutoML integration update,
- and multiple documentation/navigation/deployment improvements.
