import pickle
import threading
import sympy as sp
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import trimesh
import shapely
from shapely.validation import explain_validity
import trimesh.repair
import scipy
from collections import Counter
import time
from scipy.spatial import cKDTree
from itertools import combinations
import plotly.graph_objects as go
import dash
from dash import Dash, dcc, html, Input, Output, State
from trimesh.transformations import rotation_matrix
import itertools
import pickle as pkl
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tqdm import tqdm
from matplotlib.widgets import Slider
from scipy.optimize import minimize, root
from shapely.geometry import Polygon


print(trimesh.__version__)


#leme, patilhao, mastro, vela meshes





class Boat:
    def __init__(self, l0, l1, l2, hx, hz, o_deg, R=100):
        self.decimals = 8
        self.l0 = l0
        self.l1 = l1
        self.l2 = l2
        self.hx = hx
        self.hz = hz
        self.R = R
        self.o = np.radians(o_deg)

        self.key = (l0,l1,l2,hx,hz,o_deg,R)
        self.mesh_cache = {}
        self.volume_mesh_rot_cache = {}

        self.r1, self.a, self.h0, self.h2 = self._boat_base_solver()
        self.r2 = self.hz * np.tan(self.o) + self.r1
        self.l = self.hz / np.cos(self.o)
        self.h20 = self.cone_surface_y(self.l0, self.hz)
        self.h20x = self.l0
        #h20 belongs [self.cone_surface_x(self.h0, self.hz),self.cone_surface_y(self.l0, self.hz)]

        self.l11 = self.cone_surface_x(0, self.hz)
        self.l22 = self.cone_surface_x(self.h2, self.hz)

        self.phi0 = np.arctan(self.h0 / (self.l0 + self.a))
        self.phi2 = np.arctan(self.h2 / (self.l2 + self.a))
        self.phi02 = np.arctan(self.h20 / (self.l0 + self.a))
        self.phi22 = np.arctan(self.h2 / (self.l22 + self.a))

        self.psi = 2 * np.pi * (1 - (self.r2 - self.r1) * np.cos(self.o) / self.hz)

        self.h3 = self.h20 / 3
        self.l3 = self.l1
        self.l4 = (self.l1+self.l0)/2
        self.line_lengts = {}
    def calculate_line_length(self, x, y):
        # Ensure x and y are numpy arrays
        x = np.array(x)
        y = np.array(y)

        # Calculate the differences between consecutive points
        dx = np.diff(x)
        dy = np.diff(y)

        # Calculate the distance between consecutive points
        distances = np.sqrt(dx ** 2 + dy ** 2)

        # Sum the distances to get the total length of the line
        length = np.sum(distances)

        return length

    def cone_surface_x(self, y, z, branch='right'):
        Rz = self.radi_at_height(z)
        inside = Rz ** 2 - np.array(y) ** 2
        inside = np.clip(inside, 0.0, None)
        root = np.sqrt(inside)
        return -self.a + root

    def cone_surface_y(self, x, z):
        Rz = self.radi_at_height(z)
        inside = Rz ** 2 - (x + self.a) ** 2
        inside = np.clip(inside, a_min=0.0, a_max=None)
        return np.sqrt(inside)
    def cone_surface_z(self, x, y):
        return (self.hz / (self.r2 - self.r1)) * (-self.r1 + np.sqrt((x + self.a) ** 2 + y ** 2))
    def radi_at_height(self, z):
        return (z/self.hz)*(self.r2-self.r1)+self.r1

    def phi_f(self, y, x):
        return np.arctan(y/(x+self.a))
    def _boat_base_solver(self):
        h1 = 0.0

        # Precompute common terms for quadratic in 'a'
        c = self.hx ** 2 + self.l0 ** 2 - self.l2 ** 2
        d = 4 * (self.l2 - self.l0) ** 2
        f = 4 * c * (self.l2 - self.l0) - 8 * self.hx ** 2 * (self.l2 - self.l1)
        g = c ** 2 - 4 * self.hx ** 2 * (self.l1 ** 2 - self.l2 ** 2)

        # Discriminant
        disc = f ** 2 - 4 * d * g
        if disc < 0:
            raise ValueError(f"Negative discriminant: {disc}")

        # Two possible solutions for 'a'
        sqrt_disc = np.sqrt(disc)
        a1 = -(-f + sqrt_disc) / (2 * d)
        a2 = -(-f - sqrt_disc) / (2 * d)

        # Choose the physically meaningful root (e.g., positive shift)
        a = a1 if a1 >= 0 else a2

        # Compute base radius and heights
        r = self.l1 + a
        h0 = h1 + np.sqrt(r ** 2 - (self.l0 + a) ** 2)
        h2 = h0 - self.hx

        return r, a, h0, h2
    def r_unfold(self, r):
        return (2 * np.pi / (2 * np.pi - self.psi)) * r
    def o_unfold(self, o):
        return (1 - self.psi / (2 * np.pi)) * o

    def make_unfolded_side_panel(self, rs, phis):
        phi_ = self.o_unfold(phis)
        rs_ = self.r_unfold(rs)
        side_x = rs_ * np.cos(phi_)
        side_y = rs_ * np.sin(phi_)
        return side_x, side_y

    def mesh_faces(self, Nx, Ny):
        faces = []
        for i in range(Ny - 1):
            for j in range(Nx - 1):
                # Calculate vertex indices of the square
                v0 = i * Nx + j
                v1 = v0 + 1
                v2 = v0 + Nx
                v3 = v2 + 1

                # Two triangles per quad
                faces.append([v0, v1, v2])
                faces.append([v1, v3, v2])
        return np.array(faces)
    def mirror_line_mesh_faces(self, N):
        faces = []
        for i in range(N - 1):
            # Indices for quad: [left_i, left_i+1, right_i], [right_i, left_i+1, right_i+1]
            li = i
            li1 = i + 1
            ri = i + N
            ri1 = i + N + 1
            faces.append([li, li1, ri])
            faces.append([ri, li1, ri1])
        faces = np.array(faces)
        return faces
    def base_panel(self):
        #Line3D
        if 1:
            y = np.linspace(self.h2, self.h0, self.R)
            x = -self.a + np.sqrt(self.r1 ** 2 - y ** 2)
            boat_base_x = np.concatenate((x, [self.l0, -self.l0], -x[::-1], [-self.l2, self.l2]))
            boat_base_y = np.concatenate((y, [self.h0, self.h0], y[::-1], [self.h2, self.h2]))
            boat_base_z = np.full_like(boat_base_x, 0)
            self.base_panel_pts = np.array([boat_base_x, boat_base_y, boat_base_z]).T
            self.line_lengts['base'] ={'base':{'line3d':self.calculate_line_length(x, y)}}
        #Mesh
        if 1:
            y = np.linspace(self.h2, self.h0, self.R)
            xr = self.cone_surface_x(y,0)
            xl = -xr
            z = np.full_like(y, 0.0)
            pr = np.stack([xr, y, z], axis=1)
            pl = np.stack([xl, y, z], axis=1)
            vertices = np.concatenate([pr,pl])
            faces = self.mirror_line_mesh_faces(self.R)
            self.base_panel_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            self.base_panel_mesh.invert()

            l4, h3, h2 = self.l4, self.h3, self.h2
            vertices = np.array([[l4,h3,0], [l4,h2,0], [-l4,h2,0],[-l4,h3,0]])
            faces = np.array([[0,1,2],[2,3,0]])
            self.base_panel_inside_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            self.base_panel_inside_mesh.invert()
    def top_cover(self):
        #Line3D
        if 1:
            y = np.linspace(self.h3, self.h20, self.R)
            z = np.full_like(y, self.hz)
            x = self.cone_surface_x(y,z)
            top_cover_x = np.concatenate([x, -x[::-1], [x[0]]])
            top_cover_y = np.concatenate([y, y[::-1], [y[0]]])
            top_cover_z = np.full_like(top_cover_x, self.hz)
            self.top_cover_large_pts = np.array([top_cover_x, top_cover_y, top_cover_z]).T
        #Mesh
        if 1:
            xr = x
            xl = -xr
            z = np.full_like(y, self.hz)
            vertices = np.concatenate([np.stack([xr, y, z], axis=1), np.stack([xl,y,z], axis=1)])
            faces = self.mirror_line_mesh_faces(self.R)
            self.top_cover_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            self.top_cover_mesh.invert()
    def full_top_cover_mesh_f(self):
        y = np.linspace(self.h2, self.h20, self.R)
        xr = self.cone_surface_x(y, self.hz)
        xl = -xr
        z = np.full_like(y, self.hz)
        vertices = np.concatenate([np.stack([xr, y, z], axis=1), np.stack([xl, y, z], axis=1)])
        faces = self.mirror_line_mesh_faces(self.R)
        self.complete_top_cover_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        #self.complete_top_cover_mesh.invert()
    def top_covers_small(self):
        #Line3D
        if 1:
            y = np.linspace(self.h2, self.h3, self.R)
            z = np.full_like(y, self.hz)
            x = self.cone_surface_x(y,z)

            # Right side plate (positive x direction)
            small_right_x = np.concatenate([x, np.full_like(x, self.l3), [x[0]]])
            small_right_y = np.concatenate([y, y[::-1], [y[0]]])
            small_right_z = np.full_like(small_right_x, self.hz)

            self.top_cover_small_pts = np.column_stack([small_right_x, small_right_y, small_right_z])
        #Mesh
        if 1:
            xr = x
            xl = np.full_like(y, self.l3)
            z = np.full_like(y, self.hz)
            vertices = np.concatenate([np.stack([xr, y, z], axis=1), np.stack([xl, y, z], axis=1)])
            faces = self.mirror_line_mesh_faces(self.R)
            self.top_cover_small_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    def back_panel(self):
        #Line3D
        if 1:
            z = np.linspace(0, self.hz, self.R)
            y = np.full_like(z, self.h2)
            x = self.cone_surface_x(y, z)
            back_panel_x = np.concatenate([x, -x[::-1], [x[0]]])
            back_panel_y = np.full_like(back_panel_x, self.h2)
            back_panel_z = np.concatenate([z, z[::-1], [z[0]]])
            self.back_panel_pts = np.array([back_panel_x, back_panel_y, back_panel_z]).T
            self.line_lengts['back'] = {'back':{'line3d':self.calculate_line_length(x,z)}}
        #unfolded
        if 1: self.unfolded_back_panel_pts = np.array([back_panel_x, back_panel_z]).T
        #Mesh
        if 1:
            xr = x
            xl = -xr
            vertices = np.concat([np.stack([xr, y, z], axis=1), np.stack([xl, y, z], axis=1)])
            faces = self.mirror_line_mesh_faces(self.R)
            self.back_panel_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    def mid_panel(self):
        #Line3D
        if 1:
            l3, l4, h2, h3, hz = self.l3, self.l4, self.h2, self.h3, self.hz
            x = [-l4,l4,l3,-l3,-l4]
            y = [h3]*len(x)
            z = [0, 0, hz, hz, 0]
            self.mid_panel_pts = np.array([x, y, z]).T
        #Mesh
        if 1:
            vertices = self.mid_panel_pts
            faces = np.array([[0,1,2], [2,3,4]])
            self.mid_panel_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    def front_panel(self):
        #3D line
        if 1:
            y = np.linspace(self.h0, self.h20, self.R)
            if self.l0 == self.h20x:
                x = np.full_like(y, self.l0)
            else:
                x = np.linspace(self.l0, self.h20x, self.R)
            z = self.cone_surface_z(x, y)

            self.k0 = self.calculate_line_length(y,z)
            self.line_lengts['front'] = {'front':{'line3d':self.k0}}
            front_panel_x = np.concatenate([x, -x, [x[0]]])
            front_panel_y = np.concatenate([y, y[::-1], [y[0]]])
            front_panel_z = np.concatenate([z, z[::-1], [z[0]]])
            self.front_panel_pts = np.array([front_panel_x, front_panel_y, front_panel_z]).T
        #unfolded
        if 1:
            front_x = np.array([self.l0, self.l0, -self.l0, -self.l0, self.l0])
            front_y = np.array([0, self.k0, self.k0, 0, 0]) + self.h0
            self.unfolded_front_panel_pts = np.array([front_x, front_y]).T
        # Mesh
        if 1:
            Nz = self.R
            z = np.linspace(0, self.hz, Nz)
            if self.l0 == self.h20x:xr = np.full_like(z, self.l0)
            else: xr = np.linspace(self.l0,self.h20x,self.R)

            xl = -xr
            y = self.cone_surface_y(xr, z)
            vertices = np.concat([np.stack([xr, y, z], axis=1), np.stack([xl, y, z], axis=1)])
            faces = self.mirror_line_mesh_faces(Nz)
            self.front_panel_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            self.front_panel_mesh.invert()



    def side_panels(self):
        h3, h20, hz = self.h3, self.h20, self.hz
        h2, h0 = self.h2, self.h0
        #3D lines
        if 1:
            y1 = np.linspace(self.h2, self.h20, self.R)
            x1 = self.cone_surface_x(y1, self.hz)
            self.line_lengts['top'] = {'side panel':{'line3d':self.calculate_line_length(x1, y1)}}
            large_x = np.concatenate([x1, -x1[::-1], [x1[0]]])
            large_y = np.concatenate([y1, y1[::-1], [y1[0]]])
            large_z = np.full_like(large_x, self.hz)
            self.side_panel_large_pts = np.array([large_x, large_y, large_z]).T

        #Unfolded
        if 1:
            r = np.linspace(self.r1, self.r2, self.R)

            phi_b = np.arcsin(self.h2 / r)
            phi_r = np.linspace(self.phi22, self.phi02, self.R)
            phi_t = np.arccos((self.l0 + self.a) / r[::-1])
            phi_l = np.linspace(self.phi0, self.phi2, self.R)
            phi_n = [phi_b,phi_r,phi_t,phi_l]
            phi = np.concatenate(phi_n)
            r_b = r
            r_r = np.full((self.R,), self.r2)
            r_t = r[::-1]
            r_l = np.full((self.R,), self.r1)
            r_n = np.array([r_b, r_r, r_t, r_l])
            rs = np.concatenate(r_n)

            side_x, side_y = self.make_unfolded_side_panel(rs, phi)
            self.unfolded_side_panel_pts = np.array([side_x, side_y]).T

            keys = ['top','base', 'back','front']
            pars = [(r_r, phi_r), (r_l, phi_l), (r_b, phi_b),(r_t, phi_t)]
            for i in range(len(keys)):
                side_x, side_y = self.make_unfolded_side_panel(*pars[i])
                lenght = self.calculate_line_length(side_x, side_y)
                if self.line_lengts[keys[i]].get('side panel', None) == None:self.line_lengts[keys[i]]['side panel'] = {'unfold':lenght}
                else: self.line_lengts[keys[i]]['side panel']['unfold'] = lenght

            if 0:

                for i, (r_i, phi_i) in enumerate(zip(r_n, phi_n)):
                    phi_ = self.o_unfold(phi_i)
                    rs_ = self.r_unfold(r_i)
                    side_x = rs_ * np.cos(phi_)
                    side_y = rs_ * np.sin(phi_)
                    plt.plot(side_x, side_y, label=i)
                plt.legend()
                plt.show()

        #Mesh
        if 1:
            Ny = self.R
            Nz = self.R

            z = np.linspace(0, hz, Nz)
            if self.l0 == self.h20x: x = np.full_like(z, self.l0)
            else: x = np.linspace(self.l0, self.h20x, self.R)
            vertices = []
            for xi,zi in zip(x,z):
                hzi = self.cone_surface_y(xi, zi)
                y = np.linspace(h2, hzi, Ny)
                for yi in y:
                    xi = self.cone_surface_x(yi,zi)
                    vertices += [[xi, yi, zi]]
            vertices = np.array(vertices)
            faces = self.mesh_faces(Ny, Nz)
            self.side_panel_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    def small_inside_panels(self):
        #Line3D
        if 1:
            l3, l4, h2, h3, hz =  self.l3, self.l4, self.h2, self.h3, self.hz
            x = [l4,l4,l3,l3,l4]
            y = [h2, h3, h3, h2, h2]
            z = [0, 0, hz, hz, 0]
            self.small_inside_panels_pts = np.array([x,y,z]).T
        #Unfolded
        if 1:
            side = np.sqrt((l4-l3)*(l4-l3)+hz*hz)
            x = [0,side,side,0,0]
            y = [0,0,h3-h2, h3-h2,0]
            self.small_inside_panels_unfolded_pts = np.array([x,y]).T
        #Mesh
        if 1:
            vertices = self.small_inside_panels_pts
            faces = np.array([[0,1,2],[2,3,0]])
            self.small_inside_panels_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            self.small_inside_panels_mesh.invert()

    def top_hole_mesh_f(self):
        l3,h3,h2,hz = self.l3, self.h3, self.h2, self.hz
        vertices = np.array([[l3, h3, hz], [l3, h2, hz], [-l3, h2, hz], [-l3, h3, hz]])
        faces = np.array([[0,1,2], [2,3,0]])
        self.top_hole_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        self.top_hole_mesh.invert()
    def generate_all_geometry(self):
        self.base_panel()

        self.top_cover()
        self.top_covers_small()
        self.full_top_cover_mesh_f()
        self.top_hole_mesh_f()

        self.back_panel()
        self.mid_panel()
        self.front_panel()

        self.side_panels()
        self.small_inside_panels()


        volume_meshes = [self.base_panel_mesh, self.front_panel_mesh, self.back_panel_mesh,
                  self.complete_top_cover_mesh]
        volume_mirror_meshes = [self.side_panel_mesh]
        for mirror_mesh in volume_mirror_meshes:
            mirrored_mesh = mirror_mesh.copy()
            mirrored_mesh.apply_scale([-1, 1, 1])
            volume_meshes += [mirror_mesh, mirrored_mesh]

        self.volume_meshes = volume_meshes
        self.volume_boat_mesh = trimesh.util.concatenate(self.volume_meshes)
        self.volume_boat_mesh.merge_vertices(digits_vertex=3)

        meshes = [self.base_panel_mesh, self.front_panel_mesh, self.mid_panel_mesh, self.back_panel_mesh,
                  self.top_cover_mesh]
        mirror_meshes = [self.side_panel_mesh, self.small_inside_panels_mesh, self.top_cover_small_mesh]
        for mirror_mesh in mirror_meshes:
            mirrored_mesh = mirror_mesh.copy()
            mirrored_mesh.apply_scale([-1, 1, 1])
            # mirrored_mesh.invert()
            meshes += [mirror_mesh, mirrored_mesh]
        self.meshes = meshes
        self.boat_mesh = trimesh.util.concatenate(self.meshes)
        self.boat_mesh.merge_vertices(digits_vertex=3)

    def Plot3D(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_title('3D Boat Model')
        ax.grid(True)
        ax.legend()

        #self.generate_all_geometry()

        L = 2.2
        y_offset = 0.3
        ax.set_xlim(-L / 2, L / 2)
        ax.set_ylim(-L / 2 + y_offset, L / 2 + y_offset)
        ax.set_zlim(-L / 2, L / 2)

        # Use the stored geometry points
        ax.plot(self.small_inside_panels_pts[:,0],self.small_inside_panels_pts[:,1],self.small_inside_panels_pts[:,2], color='blue')
        ax.plot(-self.small_inside_panels_pts[:, 0], self.small_inside_panels_pts[:, 1], self.small_inside_panels_pts[:, 2], color='blue')

        ax.plot(self.base_panel_pts[:, 0], self.base_panel_pts[:, 1], self.base_panel_pts[:, 2], color='blue')

        ax.plot(self.top_cover_large_pts[:, 0], self.top_cover_large_pts[:, 1], self.top_cover_large_pts[:, 2],color='purple')

        ax.plot(-self.top_cover_small_pts[:, 0], self.top_cover_small_pts[:, 1],self.top_cover_small_pts[:, 2], color='purple')
        ax.plot(self.top_cover_small_pts[:, 0], self.top_cover_small_pts[:, 1],self.top_cover_small_pts[:, 2], color='purple')

        ax.plot(self.back_panel_pts[:, 0], self.back_panel_pts[:, 1], self.back_panel_pts[:, 2], color='green')
        ax.plot(self.mid_panel_pts[:,0],self.mid_panel_pts[:,1],self.mid_panel_pts[:,2], color='green')
        ax.plot(self.front_panel_pts[:, 0], self.front_panel_pts[:, 1], self.front_panel_pts[:, 2], color='green')
        return fig, ax
    def Plot3DExplain(self):
        fig, ax = self.Plot3D()
        # Annotated points
        key_pts = {
            #"Base center r1": np.array([-self.a, 0, 0]),
           # "Top center r2": np.array([-self.a, 0, self.hz]),
            "(l0, h0, 0)": np.array([self.l0, self.h0, 0]),
            "(l2, h2, 0)": np.array([self.l2, self.h2, 0]),
            "(h20x, h20, hz)": np.array([self.h20x, self.h20, self.hz]),
            "(l22, h2, hz)": np.array([self.l22, self.h2, self.hz]),
            "(c(h3, hz), h3, hz)": np.array([self.cone_surface_x(self.h3, self.hz), self.h3, self.hz]),
            "(l1,0,0)":np.array([self.l1,0,0]),
            "(-l3,h3,hz)":np.array([-self.l3, self.h3, self.hz]),
            "(-l4,h3,0)":np.array([-self.l4, self.h3, 0]),
        }

        if 1:
            for name, pt in key_pts.items():
                ax.scatter(*pt, color='red')
                ax.text(*pt, name, fontsize=9)

        offset = 0.02
        points_list = [[(-self.l11,0, self.hz),(self.l11,0, self.hz)]]
        for p1,p2 in points_list:
            ax.plot([p1[0],p2[0]],[p1[1],p2[1]], [p1[2],p2[2]], 'b--')
            if p1[0] == p2[0]:
                ax.text(p1[0]+offset, (p1[1]+p2[1])/2, 0, f"L = {np.round(np.abs(p1[1]-p2[1]),2)}", color='blue')
            else:
                ax.text(p1[1] + offset, (p1[0] + p2[0]) / 2, 0, f"L = {np.round(np.abs(p1[0] - p2[0]),2)}", color='blue')

        ax.plot([self.l0, self.l0], [self.h2, self.h0], [0, 0], 'b--')
        ax.text(self.l0 + 0.02, (self.h0 + self.h2) / 2, 0, f"hx = {self.hx}", color='blue')

    def PlotParts(self):
        #self.generate_all_geometry()

        fig, ax = plt.subplots(1,1)

        x,y = self.base_panel_pts[:, 0], self.base_panel_pts[:, 1]
        ax.plot(x,y)
        x, y = self.unfolded_side_panel_pts[:, 0], self.unfolded_side_panel_pts[:, 1]

        x01, _ = self.make_unfolded_side_panel(self.r1, 0)
        x02, _ = self.make_unfolded_side_panel(self.r2, 0)
        side_panel_width = x02 - x01

        x0 = self.l1-x01
        x += x0
        ax.plot(x, y)
        ax.plot(-x, y)


        x,y = self.unfolded_back_panel_pts[:, 0], self.unfolded_back_panel_pts[:, 1]
        y += np.min(self.unfolded_side_panel_pts[:,1])-self.hz
        ax.plot(x,y)
        x,y = self.unfolded_front_panel_pts[:, 0], self.unfolded_front_panel_pts[:, 1]
        ax.plot(x,y)


        x0 = self.l11 + self.l1 + side_panel_width

        x,y = self.top_cover_large_pts[:,0],self.top_cover_large_pts[:,1]
        x += x0
        ax.plot(x,y)
        x, y = self.top_cover_small_pts[:, 0], self.top_cover_small_pts[:, 1]
        x += x0
        ax.plot(x, y)
        x -= 2*x0
        ax.plot(-x, y)


        x,y = self.mid_panel_pts[:,0], self.mid_panel_pts[:,2]
        x += x0
        y += self.h3-self.hz
        ax.plot(x,y)

        x,y = self.small_inside_panels_unfolded_pts[:,0],self.small_inside_panels_unfolded_pts[:,1]
        y0 = -max(y)+self.h3-self.hz
        y += y0
        x+=x0
        ax.plot(x,y)
        x-=2*x0
        ax.plot(-x, y)


        ax.grid(True)
#        ax.gca().set_aspect('equal', adjustable='box')
        ax.set_title("Unfolded Parts")
        #plt.show()

    def find_open_boundary_edges(self, mesh: trimesh.Trimesh):
        """
        Return open (boundary) edges of a mesh.
        An edge is boundary if it is referenced by only one face.
        """
        # Get all face edges
        edges = np.sort(np.vstack([
            mesh.faces[:, [0, 1]],
            mesh.faces[:, [1, 2]],
            mesh.faces[:, [2, 0]]
        ]), axis=1)

        # Count edge occurrences
        edge_list = [tuple(edge) for edge in edges]
        edge_counts = Counter(edge_list)

        # Boundary edges appear only once
        boundary_edges = np.array([edge for edge, count in edge_counts.items() if count == 1])

        return boundary_edges

    def plot_mesh_with_boundaries(self,mesh: trimesh.Trimesh):
        """
        Visualize the mesh and its open boundaries (in red).
        """
        scene = trimesh.Scene()
        scene.add_geometry(mesh)

        # Find boundary edges
        boundary_edges = self.find_open_boundary_edges(mesh)
        if len(boundary_edges) > 0:
            # Create lines for boundary edges
            lines = trimesh.load_path(mesh.vertices[np.array(boundary_edges)])
            lines.colors = np.tile([255, 0, 0, 255], (len(lines.entities), 1))  # Red RGBA
            scene.add_geometry(lines)
            print(f"Found {len(boundary_edges)} boundary edges.")
        else:
            print("No boundary edges found. Mesh is watertight.")

        scene.show()

    def PlotMeshes(self):
        #self.generate_all_geometry()
        axis = trimesh.creation.axis(origin_size=0.02)
        scene = trimesh.Scene()
        scene.add_geometry(axis)

        for i,mesh in enumerate(self.meshes):

            flipped = mesh.copy()
            flipped.invert()
            mesh.visual.face_colors = [0, 0, 255, 255] # blue
            flipped.visual.face_colors = [255, 0, 0, 255] # red
            scene.add_geometry(mesh)
            scene.add_geometry(flipped)


        scene.show()

    def PlotMesh(self, mesh):
        axis = trimesh.creation.axis(origin_size=0.02)
        scene = trimesh.Scene()
        scene.add_geometry(axis)

        scene.add_geometry(mesh)
        scene.show()
    def inertia_y(self, contour):
        # contour é uma lista de (x, y) pontos no plano horizontal
        x, y = np.array(contour).T
        n = len(x)
        I = 0
        for i in range(n):
            j = (i + 1) % n
            cross = x[i] * y[j] - x[j] * y[i]
            I += cross * (x[i] ** 2 + x[i] * x[j] + x[j] ** 2)
        return abs(I) / 12

    def inertia_x(self, contour):
        # contour é uma lista de (x, y) pontos no plano horizontal
        x, y = np.array(contour).T
        n = len(x)
        I = 0
        for i in range(n):
            j = (i + 1) % n
            cross = x[i] * y[j] - x[j] * y[i]
            I += cross * (y[i] ** 2 + y[i] * y[j] + y[j] ** 2)
        return abs(I) / 12

    def __str__(self):
        str = f'Boat properties:\nWeight: {self.boat_weight}\nBuoy Impulsion Kg: {self.buoy_volume*self.water_density}\nCabin Weight: {self.cabin_volume*self.water_density}\nCabin width top: {self.cabin_width_top}\nCabin width bottom: {self.cabin_width_bottom}\nLenght bottom: {self.lenght_bottom}\nLenght top: {self.lenght_top}\nBeam: {self.beam}\nBottom width: {self.l1*2}\n2*l22: {self.l22 * 2}'
        return str
    def BoatProperties(self):
        self.wood_density = 500
        self.water_density = 1000
        self.g = 9.81

        surface_area = self.boat_mesh.area
        self.wood_volume = surface_area * 0.005
        self.boat_weight = self.wood_volume * self.wood_density

        self.CG = self.boat_mesh.centroid

        self.lenght_bottom = self.hx
        self.lenght_top = self.h20 - self.h2
        self.beam = self.l11 * 2
        self.buoy_volume = self.boat_mesh.volume
        self.cabin_volume = 0.5*(self.l4+self.l3)*self.hz*(self.h3-self.h2)
        self.cabin_width_bottom = self.l4*2
        self.cabin_width_top = self.l3*2
        # LPP ?
        #Draft ? falta o patilhao.. min and max,
        # distancia vertical entre linha de agua e ponto mais baixo do casco
        #mesh and z waterline
        if 1:
            Force_application_point_ = np.array([self.CG])
            Force_vectors_ = np.array([np.array([0, 0, -self.boat_weight * self.g])])
#            self.DynamicBoatProperties(Force_application_point_, Force_vectors_)
    def DynamicBoatProperties(self, Force_application_point_, Force_vectors_):
        θx_opt, θy_opt = self.FindEquilibrium(Force_application_point_, Force_vectors_)
        Force_application_point, Force_vectors, Torque, z_waterline = self.TorqueF((θx_opt, θy_opt), Force_application_point_, Force_vectors_)
        # mesh
        if 0:
            hzi = self.cone_surface_y(self.l0, z_waterline)
            y_waterline = np.linspace(self.h2, hzi, self.R)
            x_waterline = self.cone_surface_x(y_waterline, z_waterline)
            z = np.full_like(y_waterline, z_waterline)
            pr = np.stack([x_waterline, y_waterline, z]).T
            pl = pr.copy()
            pl[:, 0] = -pl[:, 0]
            vertices = np.concatenate([pr, pl], axis=0)
            contour = np.concatenate([pr, pl[::-1, :]], axis=0)
            faces = self.mirror_line_mesh_faces(self.R)

        if 1:
            waterplane = self.waterplane(self.volume_boat_mesh, z_waterline)
            x_waterline = waterplane.vertices[:,0]
            y_waterline = waterplane.vertices[:, 1]

        CB = Force_application_point[-1]
        I = Force_vectors[-1][2]
        V = I / (self.g * self.water_density)
        Iwl = self.inertia_y(waterplane.vertices[:, :2])
        BM = Iwl / V
        BG = (self.CG - CB)[2]

        GM = BM - BG
        z_waterline = z_waterline

        lenght_water_line = np.max(y_waterline) - np.min(y_waterline)
        width_water_line = np.max(x_waterline) - np.min(x_waterline)
        freeboard = self.hz - z_waterline
        displacement = self.boat_weight  # missing max!

    def rot_translate(self, angle_x_deg, angle_y_deg, center_rot=np.array([0,0,0])):
        angle_y = np.deg2rad(angle_y_deg)
        angle_x = np.deg2rad(angle_x_deg)
        yaw_axis = [0, 1, 0]  # Yaw around Y-axis
        pitch_axis = [1, 0, 0]  # Pitch around X-axis

        # Create rotation matrices
        rot_yaw = trimesh.transformations.rotation_matrix(angle_y, yaw_axis)
        rot_pitch = trimesh.transformations.rotation_matrix(angle_x, pitch_axis)
        combined_rotation = np.dot(rot_yaw, rot_pitch)
        T = trimesh.transformations.translation_matrix(-center_rot)
        T_inv = trimesh.transformations.translation_matrix(center_rot)
        final_transform = np.dot(np.dot(T_inv, combined_rotation), T)
        return final_transform
    def mesh_rot_cache_f(self, angles_x, angles_y, center_rot=np.array([0,0,0])):
        mesh_ = self.volume_boat_mesh
        top_mesh_ = self.top_hole_mesh
        for angle_y_deg in angles_y:
            for angle_x_deg in angles_x:
                if self.volume_mesh_rot_cache.get((angle_x_deg, angle_y_deg), None) == None:
                    final_transform = self.rot_translate(angle_x_deg, angle_y_deg, center_rot)
                    mesh = mesh_.copy()
                    top_mesh = top_mesh_.copy()
                    mesh.apply_transform(final_transform)
                    top_mesh.apply_transform(final_transform)
                    self.volume_mesh_rot_cache[(angle_x_deg, angle_y_deg)] = (mesh, top_mesh)

    def find_equilibrium_z_backup(self, mesh, W, z_low, z_high, tol=1e-4, max_iter=50):
        rho = self.water_density
        g = self.g
        """
        Find z such that F_buoyancy(z) + W = 0 by bisection.

        Parameters
        ----------
        mesh : trimesh.Trimesh
            The already-rotated hull mesh.
        W : float
            Weight (negative buoyancy force), i.e. -boat_weight * g.
        rho : float
            Water density.
        g : float
            Gravitational acceleration.
        z_low, z_high : float
            Initial bracket where F_net(z_low) < 0 and F_net(z_high) > 0.
        tol : float
            Desired precision in z (meters).
        max_iter : int
            Maximum bisection iterations.

        Returns
        -------
        z_eq : float
            Approximate equilibrium immersion depth.
        """

        def net_force_z(z):
            # slice hull at plane z, compute buoyant force
            plane_origin = [0, 0, z]
            plane_normal = [0, 0, -1]
            submerged = mesh.slice_plane(plane_origin, plane_normal, cap=True)
            V = submerged.volume
            F_b = rho * g * V  # upward buoyant force
            return F_b + W  # positive if net up, negative if sinking

        F_low = net_force_z(z_low)
        F_high = net_force_z(z_high)
        if F_low > 0 or F_high < 0:
            raise ValueError("Bad initial bracket: need F_low<0<F_high")

        for i in range(max_iter):
            z_mid = 0.5 * (z_low + z_high)
            F_mid = net_force_z(z_mid)

            # if exactly zero (or within tol), done
            if abs(F_mid) < 1e-6:
                return z_mid

            # pick the half‐interval that still brackets the root
            if F_mid > 0:
                # net upward at mid ⇒ root is below
                z_high = z_mid
                F_high = F_mid
            else:
                # net downward at mid ⇒ root is above
                z_low = z_mid
                F_low = F_mid

            # stop if bracket is sufficiently small
            if (z_high - z_low) < tol:
                return 0.5 * (z_low + z_high)

        # if we get here, return midpoint anyway
        return 0.5 * (z_low + z_high)
    def find_equilibrium_z(self, mesh, W, z_low, z_high, tol=1e-4):
        rho = self.water_density
        g = self.g

        def net_force_z(z):
            plane_origin = [0, 0, z]
            plane_normal = [0, 0, -1]
            submerged = mesh.slice_plane(plane_origin, plane_normal, cap=True)
            V = submerged.volume
            F_b = rho * g * V
            return F_b + W

        # Brent's method (fast and reliable)
        z_eq = scipy.optimize.brentq(net_force_z, z_low, z_high, xtol=tol)
        return z_eq

    def waterplane(self, mesh, z):
        plane_origin = [0, 0, z]
        plane_normal = [0, 0, -1]

        # Section returns a Path3D
        waterplane_path = mesh.section(
            plane_origin=plane_origin,
            plane_normal=plane_normal
        )

        if waterplane_path is None or len(waterplane_path.entities) == 0:
            raise ValueError("Waterplane section could not be generated.")
        return waterplane_path

    def submerged_mesh_f(self, mesh, z):
        plane_origin = [0, 0, z]
        plane_normal = [0, 0, -1]
        submerged_mesh = mesh.slice_plane(plane_origin, plane_normal, cap=True)
        return submerged_mesh
    def center_flotation_f(self, mesh, z):
        waterplane_path = self.waterplane(mesh, z)
        # Convert to 2D planar coordinates
        planar, to_3D = waterplane_path.to_2D()

        # Take the largest polygon (likely the outer waterplane contour)
        polygon_2D = max(planar.polygons_full, key=lambda p: p.area)
        centroid_2D = np.array(polygon_2D.centroid.coords[0])

        # Return in 3D with z restored
        return np.array([centroid_2D[0], centroid_2D[1], z])
    def StabilitySim(self, angles_deg_xy:tuple, Force_application_point_, Force_vectors_):
        torques = np.zeros((len(angles_deg_xy[0]),len(angles_deg_xy[1]),3))
        water_level = np.zeros((len(angles_deg_xy[0]),len(angles_deg_xy[1]),1))
        water_height = np.zeros((len(angles_deg_xy[0]), len(angles_deg_xy[1]), 1))
        CGs = np.zeros((len(angles_deg_xy[0]),len(angles_deg_xy[1]),3))
        Bs = np.zeros((len(angles_deg_xy[0]),len(angles_deg_xy[1]),3))
        for i, angle_deg_x in enumerate(tqdm(angles_deg_xy[0], 'Stability sim')):
            for j, angle_deg_y in enumerate(angles_deg_xy[1]):
                ang_deg_xy = tuple([angle_deg_x, angle_deg_y])
                #mesh, top_mesh = volume_mesh_rot_cache[ang_deg_xy]
                mesh, top_mesh = self.mesh_rot_f(angle_deg_x, angle_deg_y)
                Force_application_point, Force_vectors, Torque, z = self.TorqueF(ang_deg_xy, Force_application_point_, Force_vectors_)
                CG = sum([Force_application_point[i]*Force_vectors[i][2] for i in range(len(Force_application_point)-1)])/sum([Force_vectors[i][2] for i in range(len(Force_vectors)-1)])
                B = Force_application_point[-1]
                torques[i,j] = Torque
                water_height[i,j] = z
                CGs[i,j] = CG
                Bs[i,j] = B
                top_z_min = top_mesh.bounds[0, 2]
                if z > top_z_min:
                    water_level[i,j] = 1
                else: water_level[i,j] = 0
                #print(f'θ: {ang_deg_xy}, CG: {np.round(Force_application_point[0],3)}, B: {np.round(Force_application_point[-1],3)}, T: {np.round(Torque,3)}')
        #print(torques.shape, water_level.shape)

        return angles_deg_xy, torques, water_height, water_level, CGs, Bs

    def mesh_rot_f(self, angle_x, angle_y):
        mesh = self.volume_boat_mesh.copy()
        top_mesh = self.top_hole_mesh.copy()
        final_transform = self.rot_translate(angle_x, angle_y)
        mesh.apply_transform(final_transform)
        top_mesh.apply_transform(final_transform)
        self.volume_mesh_rot_cache[(angle_x, angle_y)] = (mesh, top_mesh)
        return mesh, top_mesh
    def TorqueF(self, angle_deg_xy, Force_application_point_, Force_vectors_):
        Force_application_point = np.concatenate([Force_application_point_, np.ones((Force_application_point_.shape[0], 1))], axis=1)

        mesh, top_mesh = self.mesh_rot_f(*angle_deg_xy)
        R = self.rot_translate(*angle_deg_xy)
        Force_application_point = (R @ Force_application_point.T).T
        z_low = mesh.vertices[:, 2].min()
        z_high = mesh.vertices[:, 2].max()

        Weights = np.sum(Force_vectors_, axis=0)

        z = self.find_equilibrium_z(mesh, Weights[2], z_low, z_high, tol=1e-5)

        submerged_mesh = self.submerged_mesh_f(mesh, z)
        V = submerged_mesh.volume
        I = self.water_density * V * self.g
        Impulsion = np.array([[0, 0, I]])
        buoyancy_center = np.array([submerged_mesh.centroid])
        Force_application_point = np.concatenate([Force_application_point[:, :3], buoyancy_center])
        Force_vectors = np.concatenate([Force_vectors_, Impulsion])
        Torque = np.sum(np.cross(Force_application_point, Force_vectors), axis=0)
        return Force_application_point, Force_vectors, Torque, z

    def FindEquilibrium(self,Force_application_point_, Force_vectors_):
        def residual(angles):
            θx, θy = angles

            #print('angles', θx, θy)
            _, _, Torque, _ = self.TorqueF((θx, θy), Force_application_point_, Force_vectors_)
            #print('T', Torque[:2])
            # Torque is an array [Tx, Ty, …]; return the first two
            return Torque[:2]
        x0 = np.array([0.0, 0])
        sol = root(residual, x0, tol=1e-6)

        if not sol.success:
            raise RuntimeError(f"Root‐finding failed: {sol.message}")

        θx_opt, θy_opt = sol.x
        _, _, T, z = self.TorqueF((θx_opt, θy_opt), Force_application_point_, Force_vectors_)

        print(f"Optimal angles: θx, θy = {θx_opt}, {θy_opt}, T = {T}, z = {z}")
        return θx_opt, θy_opt
    def stability_plot(self, angles, torques, rot_axis, ax):
        ax.plot(angles, torques)
        ax.set_xlabel(f'Heel angle(º) {rot_axis}')
        ax.set_ylabel('Righting Torque (N*m)')
        ax.set_title(f'Stability Curve {rot_axis}')
        ax.grid(True)
    def stability_plots_torquexy(self, angles_deg_xy, torques_, θx_stable, θy_stable):
        fig, ax = plt.subplots(1, 2)

        θx_i = np.where(angles_deg_xy[0] == θx_stable)[0][0]
        θy_i = np.where(angles_deg_xy[1] == θy_stable)[0][0]

        angles = angles_deg_xy[0]
        torques = torques_[:, θy_i, 0].copy()
        torques[:θx_i] *= -1
        boat.stability_plot(angles, torques, 'Pitch', ax[0])

        angles = angles_deg_xy[1]
        torques = torques_[θx_i, :, 1].copy()
        torques[:θy_i] *= -1
        boat.stability_plot(angles, torques, 'Roll', ax[1])
    def stability_plot3D(self, angles_deg_xy, torques_, θx_stable, θy_stable):

        θx_i = np.where(angles_deg_xy[0] == θx_stable)[0][0]
        θy_i = np.where(angles_deg_xy[1] == θy_stable)[0][0]

        X, Y = np.meshgrid(*angles_deg_xy)
        torquesx = torques_[:, :, 0].copy()
        torquesx[:θx_i, :] *= -1
        torquesy = torques_[:, :, 1].copy()
        torquesy[:, :θy_i] *= -1

        Zx = torquesx.T
        Zy = torquesy.T

        fig = plt.figure(figsize=(12, 5))
        lw = 2
        alpha = 0.95
        zorder = 10
        # First 3D plot for Zx
        if Zy.size > 0:
            ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        else:
            ax1 = fig.add_subplot(1, 1, 1, projection='3d')
        surf1 = ax1.plot_surface(X, Y, Zx, cmap='viridis', alpha=alpha)
        fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, label='Zx Value')
        ax1.set_title('Zx Surface')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Zx')

        y0_index = np.argmin(np.abs(Y[:, 0]))  # Index where y ≈ 0
        ax1.plot3D(
            X[y0_index, :],  # all x
            Y[y0_index, :],  # y = 0
            Zx[y0_index, :],  # Z values along x
            color='red', linewidth=lw, label='y = 0', zorder=zorder
        )


        if Zy.size > 0:
            # Second 3D plot for Zy
            ax2 = fig.add_subplot(1, 2, 2, projection='3d')
            surf2 = ax2.plot_surface(X, Y, Zy, cmap='viridis', alpha=alpha)
            fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10, label='Zy Value')
            ax2.set_title('Zy Surface')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Zy')

            x0_index = np.argmin(np.abs(X[0, :]))  # Index where x ≈ 0
            ax2.plot3D(
                X[:, x0_index],  # x = 0
                Y[:, x0_index],  # all y
                Zy[:, x0_index],  # Z values along y
                color='red', linewidth=lw, label='x = 0', zorder=zorder
            )


        plt.tight_layout()
    def stability_contour_plot(self, angles_deg_xy, torques_, θx_stable, θy_stable):

        θx_i = np.where(angles_deg_xy[0] == θx_stable)[0][0]
        θy_i = np.where(angles_deg_xy[1] == θy_stable)[0][0]

        X, Y = np.meshgrid(*angles_deg_xy)
        torquesx = torques_[:, :, 0].copy()
        torquesx[:θx_i, :] *= -1
        torquesy = torques_[:, :, 1].copy()
        torquesy[:, :θy_i] *= -1

        Zx = torquesx.T
        Zy = torquesy.T

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Zx Contour
        c1 = axes[0].contourf(X, Y, Zx, cmap='viridis')
        fig.colorbar(c1, ax=axes[0], shrink=0.8, aspect=10, label='Zx Value')
        axes[0].set_title('Zx Contour')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        # Add black contour line where Zx = 0
        axes[0].contour(X, Y, Zx, levels=[0], colors='black', linewidths=1.5)

        # Zy Contour
        c2 = axes[1].contourf(X, Y, Zy, cmap='viridis')
        fig.colorbar(c2, ax=axes[1], shrink=0.8, aspect=10, label='Zy Value')
        axes[1].set_title('Zy Contour')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Y')
        # Add black contour line where Zy = 0
        axes[1].contour(X, Y, Zy, levels=[0], colors='black', linewidths=1.5)

        plt.tight_layout()

    def plot_stability_sim_visual_dash(self, volume_mesh_rot_cache, z_matrix, angles_deg_xy, CGs, Bs):
        app = dash.Dash(__name__)
        angles_x, angles_y = angles_deg_xy
        angles_x = np.array(angles_x)
        angles_y = np.array(angles_y)

        angle_x_marks = {i: f"{angles_x[i]:.1f}°" for i in range(len(angles_x))}
        angle_y_marks = {i: f"{angles_y[i]:.1f}°" for i in range(len(angles_y))}

        fixed_camera = dict(
            eye=dict(x=1.25, y=1.25, z=1.25),
            center=dict(x=0, y=0, z=0),
            up=dict(x=0, y=0, z=1)
        )

        decimals = self.decimals

        def round_key(x, y):
            return (round(x, decimals), round(y, decimals))

        def make_figure(theta_x, theta_y):
            ix = int(np.abs(angles_x - theta_x).argmin())
            iy = int(np.abs(angles_y - theta_y).argmin())
            key = round_key(angles_x[ix], angles_y[iy])

            rotated_mesh, _ = volume_mesh_rot_cache[key]
            verts = rotated_mesh.vertices
            faces = rotated_mesh.faces

            mesh = go.Mesh3d(
                x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                opacity=0.6, color='lightblue'
            )

            h = float(z_matrix[ix, iy])
            minx, maxx = verts[:, 0].min(), verts[:, 0].max()
            miny, maxy = verts[:, 1].min(), verts[:, 1].max()
            xx, yy = np.meshgrid(
                np.linspace(minx, maxx, 2),
                np.linspace(miny, maxy, 2)
            )
            zz = np.full_like(xx, h)
            plane = go.Surface(x=xx, y=yy, z=zz, opacity=0.3, showscale=False)

            scatter_cgs = go.Scatter3d(
                x=[CGs[ix,iy, 0]], y=[CGs[ix,iy,1]], z=[CGs[ix,iy, 2]],
                mode='markers',
                marker=dict(size=5, color='red', symbol='circle'),
                name='CGs'
            )

            scatter_bs = go.Scatter3d(
                x=[Bs[ix,iy, 0]], y=[Bs[ix,iy, 1]], z=[Bs[ix,iy, 2]],
                mode='markers',
                marker=dict(size=5, color='blue', symbol='square'),
                name='Bs'
            )

            fig = go.Figure(data=[mesh, plane, scatter_cgs, scatter_bs])
            fig.update_layout(
                scene=dict(
                    xaxis=dict(range=[-boat.h20, boat.h20], autorange=False),
                    yaxis=dict(range=[-boat.h20, boat.h20], autorange=False),
                    zaxis=dict(range=[-boat.h20, boat.h20], autorange=False),
                    aspectmode='cube',
                    camera=fixed_camera
                ),
                title=f"θx={angles_x[ix]:.1f}°, θy={angles_y[iy]:.1f}°, water h={h:.3f}",
                margin=dict(l=0, r=0, b=0, t=40),
                uirevision='keep-camera'
            )
            return fig

        initial_fig = make_figure(angles_x[0], angles_y[0])

        app.layout = html.Div([
            html.H3("Boat Stability 3D Visualization"),
            dcc.Graph(id='stability-graph', figure=initial_fig, style={'height': '80vh'}),
            html.Div([
                html.Label('θx (deg)'),
                dcc.Slider(
                    id='slider-x',
                    min=0,
                    max=len(angles_x),
                    step=None,
                    value=0,
                    marks=angle_x_marks,
                    included=False
                )
            ], style={'width': '80%', 'padding': '20px'}),
            html.Div([
                html.Label('θy (deg)'),
                dcc.Slider(
                    id='slider-y',
                    min=0,
                    max=len(angles_y) - 1,
                    step=None,
                    value=0,
                    marks=angle_y_marks,
                    included=False
                )
            ], style={'width': '80%', 'padding': '20px'})
        ])

        @app.callback(
            Output('stability-graph', 'figure'),
            Input('slider-x', 'value'),
            Input('slider-y', 'value')
        )
        def update_figure(ix, iy):
            theta_x = angles_x[ix]
            theta_y = angles_y[iy]
            return make_figure(theta_x, theta_y)

        server_thread = threading.Thread(target=lambda: app.run(debug=True, use_reloader=False), daemon=True)
        server_thread.start()

    def stability_sim_and_plot(self, Force_application_point_, Force_vectors_, L=100, s=20,filepath='stability_sim.pkl', load=0):
        try:
            θx_stable, θy_stable = boat.FindEquilibrium(Force_application_point_, Force_vectors_)
        except Exception as e:
            print(e)
            θx_stable, θy_stable = 0,0
        θx_range = np.arange(-L, L + s, s)
        if θx_stable not in θx_range:
            θx_range = np.round(np.sort(np.concatenate([θx_range, [θx_stable]])), boat.decimals)
        θy_range = np.arange(-L, L + s, s)
        if θy_stable not in θy_range:
            θy_range = np.round(np.sort(np.concatenate([θy_range, [θy_stable]])), boat.decimals)
        angles_deg_xy = (θx_range, θy_range)
        if not load:
            angles_deg_xy, torques_, water_height, water_level, CGs, Bs = self.StabilitySim(angles_deg_xy,Force_application_point_,Force_vectors_)
            volume_mesh_rot_cache = self.volume_mesh_rot_cache
            data_to_save = {
                'angles_deg_xy': angles_deg_xy,
                'torques': torques_,
                'water_height': water_height,
                'water_level': water_level,
                'volume_mesh_rot_cache': boat.volume_mesh_rot_cache,
                'CGs':CGs,
                'Bs':Bs
            }
            with open(filepath, 'wb') as f:
                pkl.dump(data_to_save, f)
        else:
            with open(filepath, 'rb') as f:
                loaded_data = pkl.load(f)

            angles_deg_xy = loaded_data['angles_deg_xy']
            torques_ = loaded_data['torques']
            water_height = loaded_data['water_height']
            water_level = loaded_data['water_level']
            volume_mesh_rot_cache = loaded_data['volume_mesh_rot_cache']
            CGs = loaded_data['CGs']
            Bs = loaded_data['Bs']

            # plots
        if 1:
            boat.plot_stability_sim_visual_dash(volume_mesh_rot_cache=volume_mesh_rot_cache, z_matrix=water_height,
                                                angles_deg_xy=angles_deg_xy, CGs=CGs, Bs=Bs)
            # boat.stability_plots_torquexy(angles_deg_xy, torques_, θx_stable, θy_stable)
            # boat.stability_contour_plot(angles_deg_xy, torques_, θx_stable, θy_stable)
            # boat.stability_plot3D(angles_deg_xy, torques_, θx_stable, θy_stable)
    def stability_sim_and_plot2D(self, Force_application_point_, Force_vectors_, L=100, s=20,filepath='stability_sim2D.pkl', load=0):
        if load and os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                d = pickle.load(f)
                angles_deg_x = d['angles_x']
                angles_deg_y = d['angles_y']
                water_levelx = d['water_x']
                water_levely = d['water_y']
                torquesx = d['torquesx']
                torquesy = d['torquesy']
            fig, ax = plt.subplots(1, 2)
            self.stability_plot(angles_deg_x, torquesx, 'Pitch', ax[0])
            ax[0].scatter(angles_deg_x[water_levelx], torquesx[water_levelx], label='Flood threshold')
            self.stability_plot(angles_deg_y, torquesy, 'Roll', ax[1])
            ax[1].scatter(angles_deg_y[water_levely], torquesy[water_levely], label='Flood threshold')
            
        else:
            try:
                θx_stable, θy_stable = boat.FindEquilibrium(Force_application_point_, Force_vectors_)
            except Exception as e:
                print(e)
                θx_stable, θy_stable = 0, 0
            θx_range = np.arange(-L, L + s, s)
            if θx_stable not in θx_range:
                θx_range = np.round(np.sort(np.concatenate([θx_range, [θx_stable]])), boat.decimals)
    
            θy_range = np.arange(-L, L + s, s)
            if θy_stable not in θy_range:
                θy_range = np.round(np.sort(np.concatenate([θy_range, [θy_stable]])), boat.decimals)
            angles_deg_x = (θx_range,[θy_stable])
            angles_deg_y = ([θx_stable], θy_range)
            if 1:
                angles_deg_x, torquesx_, water_height, water_levelx_, _, _ = self.StabilitySim(angles_deg_x, Force_application_point_, Force_vectors_)
    
                angles_deg_y, torquesy_, water_height, water_levely_, _, _ = self.StabilitySim(angles_deg_y,Force_application_point_,Force_vectors_)
                angles_deg_x = np.array(angles_deg_x[0])
                angles_deg_y = np.array(angles_deg_y[1])
    
                tol = 1e-6
                θx_i = np.where(np.abs(angles_deg_x-θx_stable)<tol)[0][0]
                θy_i = np.where(np.abs(angles_deg_y-θy_stable)<tol)[0][0]
    
                water_levelx_ = water_levelx_[:,0].flatten()
                water_levelx = np.concatenate([[0],np.diff(np.sign(water_levelx_))]).astype(bool)
                water_levely_ = water_levely_[0, :].flatten()
                water_levely = np.concatenate([[0],np.diff(np.sign(water_levely_))]).astype(bool)
    
                torquesx = torquesx_[:, 0, 0].copy()
                torquesx[:θx_i] *= -1
    
                torquesy = torquesy_[0, :, 1].copy()
                torquesy[:θy_i] *= -1
                fig, ax = plt.subplots(1,2)
                self.stability_plot(angles_deg_x, torquesx, 'Pitch', ax[0])
                ax[0].scatter(angles_deg_x[water_levelx], torquesx[water_levelx], label='Flood threshold')
                self.stability_plot(angles_deg_y, torquesy, 'Roll', ax[1])
                ax[1].scatter(angles_deg_y[water_levely], torquesy[water_levely], label='Flood threshold')
                with open(filepath, 'wb') as f:
                    d = {'angles_x':angles_deg_x, 'angles_y':angles_deg_y, 'water_x':water_levelx,'water_y':water_levely,'torquesx':torquesx,'torquesy':torquesy}
                    pickle.dump(d,f)

if __name__ == '__main__':

    #generate boat data
    if 0:
        boat = Boat(l0=0.4/2, l1=1/2, l2=0.8/2, hx=2, hz=0.4, o_deg=30, R=20)
        boat.generate_all_geometry()
        boat.BoatProperties()
        with open('boat_instance.pkl','wb') as f: pickle.dump(boat, f)
    else:
        with open('boat_instance.pkl', 'rb') as f: boat = pickle.load(f)

    print(boat)

    #boat line lenghts
    if 0:
        for key in boat.line_lengts:
            print(key)
            print(boat.line_lengts[key])
    #plot 3DExplain
    if 0:
       # boat.Plot3D()
        boat.Plot3DExplain()
        plt.show()
    #plot Parts
    if 0:
        boat.PlotParts()
        plt.show()

    #lightweight
    l3, l4, h3, h2, hz, h20, g = boat.l3, boat.l4, boat.h3, boat.h2, boat.hz, boat.h20,boat.g
    boat_force = np.array([0, 0, -boat.boat_weight * boat.g])
    if 0:
        Force_application_point_ = np.array([boat.CG])
        Force_vectors_ = np.array([boat_force])
        boat.stability_sim_and_plot2D(Force_application_point_, Force_vectors_, L=180, s=5)

    #1 passager
    passager_x = 0.2
    passager_y = (h2+h3)/2
    passager_z = 0.5
    passager_loc = np.array([passager_x,passager_y, passager_z])
    passager_force = np.array([0, 0, -60 * g])
    if 0:
        Force_application_point_ = np.array([boat.CG, passager_loc])
        Force_vectors_ = np.array([boat_force, passager_force])
        #boat.stability_sim_and_plot(Force_application_point_, Force_vectors_, L=90, s=30)
        boat.stability_sim_and_plot2D(Force_application_point_, Force_vectors_, L=180, s=5)

    #2 passager
    passager1_loc = np.array([passager_x, h2+0.25, passager_z])
    passager2_loc = np.array([-passager_x, h3-0.25, passager_z])
    passager1_force = np.array([0, 0, -60 * g])
    passager2_force = np.array([0, 0, -60 * g])
    if 0:
        Force_application_point_ = np.array([boat.CG, passager1_loc, passager2_loc])
        Force_vectors_ = np.array([boat_force, passager1_force, passager2_force])
        boat.stability_sim_and_plot(Force_application_point_, Force_vectors_, L=90, s=30)

    #1 passager, mast, daggerboard, rudder, weight]
    mast_loc = np.array([0, (h3+h20)/2, 2.3/2])
    mast_force = np.array([0,0,-5*g])
    daggerboard_loc = np.array([0,h3-0.15, -0.25])
    daggerboard_force = np.array([0,0,-1*g])
    rudder_loc = np.array([0,h2-0.1,0])
    rudder_force = np.array([0,0,-1*g])
    weight_loc = mast_loc + np.array([-0.2,0,0])
    weight_force = np.array([0,0,-15*g])
    if 1:
        Force_application_point_ = np.array([boat.CG, passager_loc,mast_loc,daggerboard_loc,rudder_loc, weight_loc])
        Force_vectors_ = np.array([boat_force, passager_force,mast_force,daggerboard_force,rudder_force, weight_force])
        boat.stability_sim_and_plot2D(Force_application_point_, Force_vectors_, L=90, s=1, filepath='stability_sim2D.pkl',load=1)
        boat.stability_sim_and_plot(Force_application_point_, Force_vectors_, 10, 1, filepath='stability_sim.pkl',load=0)

    plt.show()
    x=input()
