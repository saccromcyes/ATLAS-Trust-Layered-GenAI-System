# Incident Response Runbook


## Severity levels

| Severity | Definition | Response target | Status updates |
|---|---|---|---|
| SEV1 | Drones grounded fleet-wide, customer data exposed, or the cloud platform fully down | On-call engineer acknowledges the page within 5 minutes | Every 30 minutes |
| SEV2 | A major feature is unavailable for many customers, with no workaround | Acknowledge within 15 minutes | Every 60 minutes |
| SEV3 | A feature is degraded or a workaround exists | Acknowledge within 4 business hours | Daily |
| SEV4 | Cosmetic issue or minor bug | Triage in the next sprint planning | Not required |

## Roles

- **Incident Commander (IC):** owns the incident, assigns tasks, and decides when the incident is resolved. For SEV1 and SEV2 incidents, the IC must not also be the person debugging.
- **Communications Lead:** writes internal and customer-facing status updates.
- **Subject-matter experts:** engineers pulled in by the IC to investigate and fix the issue.

## First 15 minutes

1. Acknowledge the page and open an incident channel named `#inc-YYYYMMDD-short-name`.
2. Declare a severity level. When unsure, choose the higher severity; it can be downgraded later.
3. Assign an Incident Commander and a Communications Lead.
4. Post the first status update.

## Drone safety incidents

If an incident could affect drones in flight, the IC must consider issuing a fleet-wide "return to home" command. Only the IC or the Head of Flight Operations may issue this command.

## Customer data breaches

If customer data may have been exposed, the Security team and Legal must be involved immediately. Affected customers must be notified within 72 hours of the breach being confirmed.

## After the incident

A blameless postmortem is required for every SEV1 and SEV2 incident. The postmortem document must be published within 5 business days of resolution. It must include a timeline, the root cause, what went well, and action items, each with an owner and a due date.
