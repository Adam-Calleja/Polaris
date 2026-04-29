import type { DemoScenario } from "./types";

export interface AssistantPromptCard {
  scenarioId: string;
  title: string;
  description: string;
  icon: string;
  prompt: string;
}

type TicketExampleScenario = DemoScenario & {
  icon: string;
};

const ticketExampleScenarios: TicketExampleScenario[] = [
  {
    scenarioId: "HPCSSUP-98820",
    title: "RDS and RCS licence renewal",
    description:
      "Storage licence renewal and ownership transfer should be routed through the Self Service Storage Portal.",
    query: `Dear Storage Services team,
We would like to renew our licences for both the RDS and RCS  (account 80/81) for another year.
On top of this is it possible to change the ownership of these licences to Maria (Cc’ed) who is our new head of bioinformatics?
Many thanks
Brian
**
Dr. Brian Lam
Assistant Research Professor
Institute of Metabolic Science-Metabolic Research Laboratories
Medical Research Council Metabolic Diseases Unit
University of Cambridge
Institute of Metabolic Science
Level 4, Box 289, Addenbrooke's Hospital
Cambridge CB2 0QQ
United Kingdom
Phone: +44 (0)1223 768628
Email: yhbl2@cam.ac.uk<[yhbl2@cam.ac.uk](mailto:yhbl2@cam.ac.uk)>`,
    focus: "Shows a clean, grounded self-service answer with an explicit do-not-process-manually policy outcome.",
    includeEvaluationMetadata: true,
    icon: "✦",
  },
  {
    scenarioId: "HPCSSUP-98311",
    title: "Compiling LAMMPS on Ampere",
    description:
      "A version-sensitive GPU build problem where the response needs the exact supported Ampere module stack.",
    query: `Dear CSD3 support,
I am trying to compile the newest release version of LAMMPS on Ampere with the GPU package. Following the instructions on [https://docs.hpc.cam.ac.uk/hpc/software-packages/lammps.html](https://docs.hpc.cam.ac.uk/hpc/software-packages/lammps.html) leads to an error when running a benchmark:
Cuda driver error 1 in call at file '/home/dc-bole2/git/lammps_amp/lib/gpu/geryon/nvd_kernel.h' in line 340.
Cuda driver error 1 in call at file '/home/dc-bole2/git/lammps_amp/lib/gpu/geryon/nvd_kernel.h' in line 340.
Cuda driver error 1 in call at file '/home/dc-bole2/git/lammps_amp/lib/gpu/geryon/nvd_kernel.h' in line 340.
Cuda driver error 1 in call at file '/home/dc-bole2/git/lammps_amp/lib/gpu/geryon/nvd_kernel.h' in line 340.
--------------------------------------------------------------------------
MPI_ABORT was invoked on rank 1 in communicator MPI_COMM_WORLD
with errorcode -1.
NOTE: invoking MPI_ABORT causes Open MPI to kill all MPI processes.
You may or may not see output from other processes, depending on
exactly when Open MPI kills them.
--------------------------------------------------------------------------
This is working without any issues when using version stable_29Aug2024_update2 as per the instructions. By any chance, do you happen to have any information on how to compile a more recent version of LAMMPS?
Many thanks,
Max`,
    focus: "Demonstrates precise technical guidance for a version-sensitive support ticket without drifting into unsafe system-level advice.",
    includeEvaluationMetadata: true,
    icon: "⌁",
  },
  {
    scenarioId: "HPCSSUP-98292",
    title: "Fw: DAWN Access Assistance",
    description:
      "An access request with an SSH key pasted into the ticket that must follow the DAWN invitation and provisioning workflow.",
    query: `Dear Support,
Can someone please address this one from yesterday and today? Thanks.
Regards,
Muhammad Ahmed
________________________________
From: Jun Yao Chan <jun@revax.co.uk>
Sent: 27 January 2026 12:34
To: HPC JIRA service desk inbox <jiraservice@hpc.cam.ac.uk>
Cc: Alex Bartlam <Alex@revax.co.uk>; Jack Mander <jack@revax.co.uk>
Subject: DAWN Access Assistance
Hi,
I am looking to gain access to the DAWN Supercomputer, some of my colleagues already have access but I am not set up with an account yet. Are you able to set up login details for my account?
Here is the all the details:
Name: Jun Yao Chan
Email: jun@revax.co.uk
SSH keys: AAAAC3NzaC1lZDI1NTE5AAAAIL+klunbwQNR+EoEuzyqe0y0HGjf25GM3hIfka8DL1Ra
Thanks.
Regards,
Jun`,
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
}));
