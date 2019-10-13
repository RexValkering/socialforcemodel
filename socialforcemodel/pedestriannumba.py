from numba import jit, float32, bool_, int32
import numpy as np


@jit(float32[:](float32, float32[:], float32[:], float32, float32, float32,
     float32[:, :], float32[:, :], float32[:], float32, float32, bool_, bool_,
     float32[:], float32[:], int32))
def calculate_pedestrian_repulsive_force(distance_threshold, self_position, self_velocity,
                                         self_radius, self_speed, self_labda_scale, ped_position, ped_velocity,
                                         ped_radius, world_height, world_width, continuous_domain,
                                         ignore_pedestrians_behind, desired_dir, force_args, k):
    """ Calculates the repulsive force with all others pedestrians. """

    social_force = np.zeros(2)
    physical_force = np.zeros(2)
    local_density = 0.0
    local_velocity_variance = 0.0
    sum_repulsive = 0.0
    sum_pushing = 0.0

    # Loop through all pedestrians.
    for i in range(len(ped_position)):

        p_position = np.array(ped_position[i])

        # Calculate the distance.
        position = self_position
        difference = p_position - position

        # In case of a continuous domain, we should check if the 'wrapped'
        # distance is closer.
        if continuous_domain:
            if difference[0] > 0.5 * world_width:
                difference[0] = difference[0] - world_width
                p_position[0] -= world_width
            elif difference[0] < - 0.5 * world_width:
                difference[0] = difference[0] + world_width
                p_position[0] += world_width

            if difference[1] > 0.5 * world_height:
                difference[1] = difference[1] - world_height
                p_position[1] -= world_height
            elif difference[1] < - 0.5 * world_height:
                difference[1] = difference[1] + world_height
                p_position[1] += world_height

        distance_squared = difference[0]**2 + difference[1]**2

        # Skip if the pedestrian is too far away. This saves a significant
        # amount of time in large groups.
        if distance_squared > distance_threshold:
            continue

        distance = np.sqrt(distance_squared)

        # Agent overlap is positive if two agents 'overlap' in space.
        agent_overlap = self_radius + ped_radius[i] - distance

        # Unit vector of the difference
        difference_direction = difference / distance

        # Find normal and tangential of difference
        normal = (position - p_position) / distance

        tangential = np.array([-normal[1], normal[0]])

        max_repulsive_force = force_args[0]
        labda = force_args[1]
        D_zero = force_args[2]
        D_one = force_args[3]
        body_force_constant = force_args[4]
        friction_force_constant = force_args[5]
        smoothing_squared = force_args[6]
        smoothing_factor = force_args[7]

        factor = max(distance, 0.15)
        cos_angle = desired_dir * difference_direction
        labda = (1.0 - self_labda_scale * (1.0 - labda))
        omega = labda + (1 - labda) * (1 + cos_angle) / 2
        social_repulsion_force = max_repulsive_force * omega * np.exp(
            - factor / D_zero + (D_one / factor)**k
        )

        pushing_force = 0
        friction_force = np.array([0, 0])

        if agent_overlap > 0 and False:
            # Find delta, which is a factor for friction force.
            delta = (ped_velocity[i] - self_velocity) * tangential

            pushing_force = body_force_constant * agent_overlap
            friction_force = (friction_force_constant * agent_overlap *
                              delta * tangential)

        # Sum the forces and add to total force.
        social_pedestrian_force = social_repulsion_force * normal
        physical_pedestrian_force = pushing_force * normal + friction_force

        social_force += social_pedestrian_force
        physical_force += physical_pedestrian_force

        pressure = smoothing_factor * np.exp(-distance_squared /
                                             smoothing_squared)

        local_density += pressure
        local_velocity_variance += self_speed * pressure

        sum_repulsive += np.sqrt(social_repulsion_force[0]**2 + social_repulsion_force[1]**2)
        sum_pushing += pushing_force

    if local_density != 0:
        local_velocity_variance /= local_density

    # print([local_density, local_velocity_variance, sum_repulsive, sum_pushing])

    return np.append(np.append(social_force, physical_force),
                    [local_density, local_velocity_variance, sum_repulsive, sum_pushing])
