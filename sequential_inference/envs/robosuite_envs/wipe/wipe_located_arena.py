import numpy as np
from robosuite.models.arenas import TableArena
from robosuite.utils.mjcf_utils import CustomMaterial, find_elements
from robosuite.models.objects import CylinderObject


class WipeLocatedArena(TableArena):
    def __init__(
        self,
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(0.01, 0.005, 0.0001),
        table_offset=(0, 0, 0.8),
        coverage_factor=0.9,
        num_markers=10,
        table_friction_std=0,
        line_width=0.02,
        num_paths=20,
        continuous_paths=True,
        location_paths="everywhere",  # everywhere, left, right, up, down
        seed=None,
    ):
        # Tactile table-specific features
        self.table_friction_std = table_friction_std
        self.line_width = line_width
        self.markers = []
        self.coverage_factor = coverage_factor
        self.num_markers = num_markers
        self.num_paths = num_paths
        self.continuous_paths = continuous_paths
        self.location_paths = location_paths
        self.seed = seed
        self.paths = []

        # run superclass init
        super().__init__(
            table_full_size=table_full_size,
            table_friction=table_friction,
            table_offset=table_offset,
        )

    def configure_location(self):
        """Configures correct locations for this arena"""
        # Run superclass first
        super().configure_location()

        # Define dirt material for markers
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.0",
            "shininess": "0.0",
        }
        dirt = CustomMaterial(
            texture="Dirt",
            tex_name="dirt",
            mat_name="dirt_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
            shared=True,
        )

        # Define line(s) drawn on table
        for i in range(self.num_markers):
            # If we're using two clusters, we resample the starting position and direction at the halfway point
            # if self.two_clusters and i == int(np.floor(self.num_markers / 2)):
            #     pos = self.sample_start_pos()
            marker_name = f"contact{i}"
            marker = CylinderObject(
                name=marker_name,
                size=[self.line_width / 2, 0.001],
                rgba=[1, 1, 1, 1],
                material=dirt,
                obj_type="visual",
                joints=None,
            )
            # Manually add this object to the arena xml
            self.merge_assets(marker)
            table = find_elements(
                root=self.worldbody,
                tags="body",
                attribs={"name": "table"},
                return_first=True,
            )
            table.append(marker.get_obj())

            # Add this marker to our saved list of all markers
            self.markers.append(marker)

            # Add to the current dirt path
            # pos = self.sample_path_pos(pos)

        self.paths = self.sample_paths(
            self.num_paths, self.continuous_paths, self.location_paths, self.seed
        )

    def reset_arena(self, sim):
        """
        Reset the visual marker locations in the environment. Requires @sim (MjSim) reference to be passed in so that
        the Mujoco sim can be directly modified
        Args:
            sim (MjSim): Simulation instance containing this arena and visual markers
        """
        # Choose new path
        path = self.paths[np.random.choice(len(self.paths))]
        # Loop through all visual markers
        for i, marker in enumerate(self.markers):
            # Get IDs to the body, geom, and site of each marker
            body_id = sim.model.body_name2id(marker.root_body)
            geom_id = sim.model.geom_name2id(marker.visual_geoms[0])
            site_id = sim.model.site_name2id(marker.sites[0])
            # Determine new position for this marker
            position = np.array([path[i][0], path[i][1], self.table_half_size[2]])
            # Set the current marker (body) to this new position
            sim.model.body_pos[body_id] = position
            # Reset the marker visualization -- setting geom rgba alpha value to 1
            sim.model.geom_rgba[geom_id][3] = 1
            # Hide the default visualization site
            sim.model.site_rgba[site_id][3] = 0

    def sample_paths(self, num_paths, continuous, location="everywhere", seed=None):
        np.random.seed(seed)
        x_min, x_max = -self.table_half_size[0], self.table_half_size[0]
        y_min, y_max = -self.table_half_size[1], self.table_half_size[1]
        if location == "everywhere":
            pass
        elif location == "left":
            y_min = 0
        elif location == "right":
            y_max = 0
        elif location == "up":
            x_min = 0
        elif location == "down":
            x_max = 0
        else:
            raise NotImplementedError("Unknown Wipe position {}".format(location))

        paths = []
        for _ in range(num_paths):
            if continuous:
                paths.append(self.sample_continuous_path(x_min, x_max, y_min, y_max))
            else:
                paths.append(self.sample_discontinuous_path(x_min, x_max, y_min, y_max))

        np.random.seed()
        return paths

    def sample_continuous_path(self, x_min, x_max, y_min, y_max):
        path = []
        pos = self.sample_start_pos(x_min, x_max, y_min, y_max)
        direction = np.random.uniform(-np.pi, np.pi)
        for _ in range(self.num_markers):
            path.append(pos)
            pos, direction = self.sample_next_pos_and_direction(
                pos, direction, x_min, x_max, y_min, y_max
            )
        return np.stack(path)

    def sample_discontinuous_path(self, x_min, x_max, y_min, y_max):
        path = []
        for _ in range(self.num_markers):
            pos = self.sample_start_pos(x_min, x_max, y_min, y_max)
            path.append(pos)
        return np.stack(path)

    def sample_start_pos(self, x_min, x_max, y_min, y_max):
        """
        Helper function to return sampled start position of a new dirt (peg) location
        Returns:
            np.array: the (x,y) value of the newly sampled dirt starting location
        """

        return np.array(
            (
                np.random.uniform(
                    x_min * self.coverage_factor + self.line_width / 2,
                    x_max * self.coverage_factor - self.line_width / 2,
                ),
                np.random.uniform(
                    y_min * self.coverage_factor + self.line_width / 2,
                    y_max * self.coverage_factor - self.line_width / 2,
                ),
            )
        )

    def sample_next_pos_and_direction(self, pos, direction, x_min, x_max, y_min, y_max):
        # Random chance to alter the current dirt direction
        if np.random.uniform(0, 1) > 0.7:
            direction += np.random.normal(0, 0.5)

        posnew0 = pos[0] + 0.005 * np.sin(direction)
        posnew1 = pos[1] + 0.005 * np.cos(direction)

        # We keep resampling until we get a valid new position that's on the table
        while (
            posnew0 <= x_min * self.coverage_factor + self.line_width / 2
            or posnew0 >= x_max * self.coverage_factor - self.line_width / 2
            or posnew1 <= y_min * self.coverage_factor + self.line_width / 2
            or posnew1 >= y_max * self.coverage_factor - self.line_width / 2
        ):
            direction += np.random.normal(0, 0.5)
            posnew0 = pos[0] + 0.005 * np.sin(direction)
            posnew1 = pos[1] + 0.005 * np.cos(direction)

        # Return this newly sampled position and direction
        return np.array((posnew0, posnew1)), direction
