# Security

## Reporting a vulnerability

Email **hasan@halacli.com** with a description and, where possible, a reproduction.
Please do not open a public issue for security problems. You will get an acknowledgement
within five working days.

## What is covered

The code on the `main` branch. Dependencies are pinned to exact versions and audited in CI
on every pull request; a known advisory fails the checks.

## Secrets

Never commit a `.env` file. `.env` is ignored by git and `.env.example` lists the
variables with placeholders, never real credentials. The CI pipeline scans the full history for committed credentials.
