#!/usr/bin/env python3
import argparse
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import List, Tuple, Dict
from xml.dom.minidom import Document
from tf_transformations import euler_from_quaternion
from copy import deepcopy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge multiple URDF/SRDF XML files with prefixes into a single XML based on a template."
    )
    parser.add_argument(
        "--template",
        required=True,
        help="Path to the template XML file containing common information (e.g., <robot> root).",
    )
    parser.add_argument(
        "--inputs",
        required=True,
        nargs="+",
        help="List of input XML files with prefixes in the form file.xml:prefix.",
    )
    parser.add_argument(
        "--output", required=True, help="Path to write the merged XML output."
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        default=[
            "link",
            "joint",
            "transmission",
            "gazebo",
            "plugin",
            "group",
            "disable_collisions",
        ],
        help="XML tags whose 'name' or other attributes should be prefixed.",
    )
    return parser.parse_args()


def prefix_attributes(
    elem: ET.Element, prefix: str, tags: List[str], attrs: List[str]
) -> None:
    if elem.tag in tags:
        for attr, val in list(elem.attrib.items()):
            if attr in attrs:
                elem.set(attr, f"{prefix}{val}")
    for child in elem:
        prefix_attributes(child, prefix, tags, attrs)


def prefix_xml(
    inputs: List[Tuple[str, str]],
    targets: List[Tuple[List[str], List[str]]],
) -> List[Tuple[str, ET.ElementTree]]:
    input_trees = []
    for prefix, content in inputs:
        tree = ET.ElementTree(ET.fromstring(content))
        root: ET.Element = tree.getroot()
        for child in list(root):
            for tags, attrs in targets:
                prefix_attributes(child, prefix + "/", tags, attrs)
        
        input_trees.append((prefix, tree))

    return input_trees


def create_origin(transform: List[str]) -> ET.Element:
    origin = ET.Element("origin")

    translate = [float(v) for v in transform[:3]]
    quaternion = [float(v) for v in transform[3:7]]
    euler = euler_from_quaternion(quaternion)
    
    origin.set("xyz", " ".join([format(v, ".6f") for v in translate]))
    origin.set("rpy", " ".join([format(v, ".6f") for v in euler]))
    return origin


def merge_urdf(
    template_path: str,
    children: List[Tuple[str, ET.ElementTree]],
    transforms: Dict[str, List[str]],
    base_link_frame: str = "base_link",
) -> ET.ElementTree:
    tpl_tree = ET.parse(template_path)
    tpl_root: ET.Element = tpl_tree.getroot()

    prefixes = [prefix for prefix, _ in children]
    children_root = [child.getroot() for _, child in children]
    
    base_link_frames = [f"{prefix}/{base_link_frame}" for prefix in prefixes]
    children_base_link = []
    for base_link_frame, child in zip(base_link_frames, children_root):
        for elem in child.findall(".//link"):
            if elem.get("name") == base_link_frame:
                children_base_link.append(elem)
                break
    
    fixed_joint = tpl_root.find(".//joint")
    tpl_root.remove(fixed_joint)

    for prefix, base_link_frame, child in zip(prefixes, base_link_frames, children_root):
        child_fixed_joint = deepcopy(fixed_joint)
        child_fixed_joint.set("name", f"{prefix}/fixed")
        child_fixed_joint.append(create_origin(transforms[prefix]))

        joint_child = child_fixed_joint.find("child")
        joint_child.set("link", base_link_frame)

        tpl_root.append(child_fixed_joint)
    
    for child_root in children_root:
        # tpl_root.append(child_root)
        for elem in child_root:
            tpl_root.append(elem)
    
    return tpl_root


def merge_srdf(
    template_path: str,
    children: List[Tuple[str, ET.ElementTree]],
) -> ET.ElementTree:
    tpl_tree = ET.parse(template_path)
    tpl_root: ET.Element = tpl_tree.getroot()

    children_root = [child.getroot() for _, child in children]
    for child_root in children_root:
        # tpl_root.append(child_root)
        for elem in child_root:
            tpl_root.append(elem)
    
    return tpl_root


def dumps_et(tree: ET.ElementTree) -> str:
    rough_str = ET.tostring(tree, "utf-8")
    reparsed: Document = minidom.parseString(rough_str)
    return reparsed.toprettyxml(indent="  ")
