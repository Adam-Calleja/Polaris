import type { DemoScenario } from "./types";

export interface AssistantPromptCard {
  scenarioId: string;
  title: string;
  description: string;
  icon: string;
  prompt: string;
  displayQuery?: string;
}

type TicketExampleScenario = DemoScenario & {
  icon: string;
};

// Fully synthetic demo scenarios. No real ticket, person, email, or key is
// represented. query and displayQuery are identical because the content is
// fabricated and needs no redaction.
const ticketExampleScenarios: TicketExampleScenario[] = [
  {
    scenarioId: "DEMO-2001",
    title: "Project storage licence renewal",
    description:
      "Storage licence renewal and ownership transfer should be routed through the self-service storage portal.",
    query: `Dear Storage Services team,
We would like to renew the storage allocation for our project (account DEMO-ACC-12) for another year.
Could ownership also be transferred to our new group lead, who is Cc'ed?
Many thanks,
A. Researcher
Group Lead, Example Research Group
researcher042@example.org`,
    displayQuery: `Dear Storage Services team,
We would like to renew the storage allocation for our project (account DEMO-ACC-12) for another year.
Could ownership also be transferred to our new group lead, who is Cc'ed?
Many thanks,
A. Researcher
Group Lead, Example Research Group
researcher042@example.org`,
    focus: "Shows a clean, grounded self-service answer with an explicit do-not-process-manually policy outcome.",
    includeEvaluationMetadata: true,
    icon: "✦",
  },
  {
    scenarioId: "DEMO-2002",
    title: "Compiling LAMMPS on the GPU partition",
    description:
      "A version-sensitive GPU build problem where the response needs the exact supported module stack.",
    query: `Dear support,
I am trying to compile the latest LAMMPS release with the GPU package on the ampere partition. Following the documentation at https://docs.hpc.cam.ac.uk/hpc/software-packages/lammps.html, the benchmark fails with:
Cuda driver error 1 in call at file 'lib/gpu/geryon/nvd_kernel.h' in line 340.
MPI_ABORT was invoked on rank 1 in communicator MPI_COMM_WORLD with errorcode -1.
The supported version stable_29Aug2024 builds fine. Do you have any guidance on compiling a more recent LAMMPS release?
Many thanks,
A. User
auser@example.org`,
    displayQuery: `Dear support,
I am trying to compile the latest LAMMPS release with the GPU package on the ampere partition. Following the documentation at https://docs.hpc.cam.ac.uk/hpc/software-packages/lammps.html, the benchmark fails with:
Cuda driver error 1 in call at file 'lib/gpu/geryon/nvd_kernel.h' in line 340.
MPI_ABORT was invoked on rank 1 in communicator MPI_COMM_WORLD with errorcode -1.
The supported version stable_29Aug2024 builds fine. Do you have any guidance on compiling a more recent LAMMPS release?
Many thanks,
A. User
auser@example.org`,
    focus: "Demonstrates precise technical guidance for a version-sensitive support ticket without drifting into unsafe system-level advice.",
    includeEvaluationMetadata: true,
    icon: "⌁",
  },
  {
    scenarioId: "DEMO-2003",
    title: "New account access request",
    description:
      "An access request with an SSH key pasted into the ticket that must follow the approved invitation and provisioning workflow.",
    query: `Dear Support,
Could you please help set up access for a new colleague? Several of our group already have accounts, but this user is not set up yet.
Here are the details:
Name: A. Newuser
Email: newuser@example.org
SSH key: ssh-ed25519 AAAAEXAMPLEONLY0000NOTAREALKEY0000EXAMPLE example-key
Thanks,
Group Admin
admin@example.org`,
    displayQuery: `Dear Support,
Could you please help set up access for a new colleague? Several of our group already have accounts, but this user is not set up yet.
Here are the details:
Name: A. Newuser
Email: newuser@example.org
SSH key: ssh-ed25519 AAAAEXAMPLEONLY0000NOTAREALKEY0000EXAMPLE example-key
Thanks,
Group Admin
admin@example.org`,
    focus: "Highlights policy-safe handling: require the approved invitation flow and avoid manual SSH key installation or account creation shortcuts.",
    includeEvaluationMetadata: true,
    icon: "⌘",
  },
];

export const demoScenarios: DemoScenario[] = ticketExampleScenarios.map(({ icon: _icon, ...scenario }) => scenario);

export const assistantPromptCards: AssistantPromptCard[] = ticketExampleScenarios.map((scenario) => ({
  scenarioId: scenario.scenarioId,
  title: scenario.title,
  description: scenario.description,
  icon: scenario.icon,
  prompt: scenario.query,
  displayQuery: scenario.displayQuery,
}));
