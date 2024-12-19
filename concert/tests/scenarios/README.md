# CI Scenarios

Continuous integration scenarios provide a way to validate expected outcomes from different
experimental setups within reasonable scope. For instance, we can develop workflows needing GPU
workload scheduling. Such scenarios can be executed locally but will not work remotely with GitHub
Actions runners. We use `docker compose` as small-scale container orchestration utility to simulate
a decentralized compute infrastructure, which concert is being developed for. Following is the
proposed semantic of scenarios.

```text
concert/tests/scenarios/
    scenario/ => Name of a given scenario.
        setup.py => Any required activity before running the scenario.
        session_*.py => Concert session where the scenario is described.
        assert.py => Assertions on the expected behavior upon running the scenario.
        teardown.py => Any cleanup activity after running the scenario.
        compose.yml => Simulates decentralized compute processes and run assertions.
```
## Container Orchestration

Simulating decentralized infrastructure is done by small-scale container orchestration, which should
be described in a yml file to use `docker compose` utility. Following is an example from remote
experiment scenario.

```yaml
services:
  uca_camera:
    container_name: uca_camera
    build: ../../../..
    restart: on-failure
    expose:
      - '8989'
      - '8993'
    command: ["ucad", "mock"]
    networks:
      concert_net:

  remote_walker:
    volumes:
      - /mnt:/mnt/ips_image_mnt
    container_name: remote_walker
    build: ../../../..
    restart: on-failure
    expose:
      - '7001'
      - '8993'
    command: ["concert", "tango", "walker", "--loglevel", "perfdebug", "--port", "7001"]
    depends_on:
      - uca_camera
    networks:
      concert_net:

  remote_experiment:
    volumes:
      - /mnt:/mnt/ips_image_mnt
      - .:/root/.local/share/concert/
    container_name: remote_experiment
    build: ../../../..
    restart: "no"
    environment:
      UCA_NET_HOST: "uca_camera"
      TARGET_LOCATION: "/mnt/ips_image_mnt"
    command: >
      bash -c "python3 /root/.local/share/concert/setup.py
      && concert start session_remote_exp 
      && python3 /root/.local/share/concert/assert.py
      && python3 /root/.local/share/concert/teardown.py"
    depends_on:
      - remote_walker
    networks:
      concert_net:

networks:
  concert_net:
```

It instructs `docker` to do the following **main** things for us, among others.

- Build an image from `concert/Dockerfile`, which encapsulates the recipe to installing libuca,
uca-net and concert from current branch.
- Create a virtual bridge network `concert_net` and run three containers (`uca_camera`,
`remote_walker`, `remote_experiment`) from built image. Compose will exclusively add these containers
to the specified network by allocating private IP addresses to them and creating DNS entries with the
same names. It means, we can resolve these domain names using the respective service names, which is particularly relevant for the concert session. Additionally, set the required environment variables
for running the session and additional Python routines.
- Containers (`uca_camera`, `remote_walker`) expose few specific ports, as needed for the concert
session, so that we can communicate with the tango server, mock uca camera and establish the stream.
- Mount files from scenario directory `.` to `/root/.local/share/concert/` for `remote_experiment`
container, so that we can call `concert start` on the session and execute other Python routines.
- Mount `/mnt` from host to `/mnt/ips_image_mnt` inside (`remote_walker`,`remote_experiment`)
so that 1) `remote_walker` container can write to it 2) `remote_experiment` can run setup, assertions
and cleanup routines.

## TODOs

- Add health-check so that experiment service deterministically waits for walker device server
removing any uncertainty.