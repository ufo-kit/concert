# CI Scenarios

Scenarios provide a way to validate expected outcomes from different measurement setups. We use
`docker compose` as small-scale container orchestration utility to simulate a decentralized compute
infrastructure, which concert is being developed for. Basic semantic for a scenario execution is to,

1. spin up a simulated decentralized infrastructure using containers running inside a virtual
private network.
2. run test and assert using `pytest`.
3. teardown the simulated decentralized infrastructure.

GitHub Actions scenarios can be executed locally using the [act](https://nektosact.com/) tool, which
in turn requires docker.

## Image

[Dockerfile](https://github.com/ufo-kit/concert/blob/master/Dockerfile) at the root of the repository
is built before scenario execution. This image specification incorporates all necessary building
blocks to run a test setup excluding GPU workload scheduling. GitHub Actions runner does not support
GPU workload scheduling for the free-tier repositories.

## Container Orchestration

Simulating decentralized infrastructure is done by small-scale container orchestration, which should
be described in a yaml file to use `docker compose` utility. Following is an example of bare minimum
yaml specification required to run an experiment.

```yaml
services:
  uca_camera:
    container_name: uca_camera
    image: concert-docker
    restart: on-failure
    expose:
      - '8989'
      - '8993'
    command: ["ucad", "mock"]
    networks:
      concert_net:

  remote_walker:
    container_name: remote_walker
    image: concert-docker
    restart: on-failure
    volumes:
      - /mnt:/mnt/ips_image_mnt
    expose:
      - '7001'
      - '8993'
    command: ["concert", "tango", "walker", "--loglevel", "perfdebug", "--port", "7001"]
    depends_on:
      - uca_camera
    networks:
      concert_net:

networks:
  concert_net:
```

It instructs `docker` to do the following before we can run tests.

- Create a virtual private bridge network `concert_net` and run two containers (`uca_camera`,
`remote_walker`) from the image built from
[Dockerfile](https://github.com/ufo-kit/concert/blob/master/Dockerfile). Compose will exclusively 
add these containers to the network by allocating private IP addresses and creating respective DNS
entries. It means, we can resolve these domain names using the respective service names within this
network. These details are important for writing a functional test case.

- Containers expose few specific ports, as needed for the test case, so that we can communicate with
the tango server, mock uca camera and establish the stream.

- Mount location `/mnt` from host to `/mnt/ips_image_mnt` inside running container as volume. Latter
is the location which would be used by the remote walker to write data. We map the directory `/mnt`
from host because this same location would also be mounted to the container running the test setup
and we need to be able to assert based on the data written to this location. Hence, this volume
mapping bridges the gap between two containers and enables us to do the necessary assertions as part
of the test case.

## TestCase Execution

As described in the [workflow](https://github.com/ufo-kit/concert/blob/master/.github/workflows/ci.yml)
a testcase is executed inside a docker container and invoked by a command like below.

```bash
docker run \
--network scenarios_concert_net \
--volume /mnt:/mnt/ips_image_mnt \
--env-file concert/tests/integration/scenarios/scenarios.env \
concert-docker \
make scenarios
```

Following aspects of this container execution are especially noteworthy.

- `--network scenarios_concert_net` specifies that the container should be attached to the same network
orchestrated by the docker compose earlier. Only then, processes running inside this container would
be able to communicate with the containers running inside that network on exposed ports. The name of
the network created by docker compose follows the convention `<directory_name>_<network_name>`, where
directory refers to the parent directory of the compose file and network name is the one specified there.

- `--volume /mnt:/mnt/ips_image_mnt` specifies that the location, where remote walker writes data,
is also available to this container so that assertions could be made by reading the written data.

- `--env-file concert/tests/integration/scenarios/scenarios.env` consolidates and injects the relevant
details of the compose orchestration to this container as environment variables, enabling us to write
a functional testcase leveraging the simulated decentralized infrastructure.

- `make scenarios` designates the command to run inside the container, which in turn executes the
test cases.
