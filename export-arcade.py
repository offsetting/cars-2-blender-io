import struct, json
from typing import Any
import numpy as np
import bpy
import binascii
from mathutils import *
from dataclasses import dataclass

FLOAT_2 = "ff" # 1
FLOAT_3 = "fff" # 2
U8_4 = 'BBBB' # 24
U8_4_NORMALIZED = 'BBBB' # 25
S10_10_10_NORMALIZED = 'I' # 31
S11_11_10_NORMALIZED = 'I' # 39
FLOAT16_2 = 'HH' # 44
FLOAT16_4 = 'HHHH' # 45
S1_6_3 = 'bbb' # 46

ID_FLOAT_2 = 1
ID_FLOAT_3 = 2
ID_U8_4 = 24
ID_U8_4_NORMALIZED = 25
ID_S10_10_10_NORMALIZED = 31
ID_S11_11_10_NORMALIZED = 39
ID_FLOAT16_2 = 44
ID_FLOAT16_4 = 45
ID_S1_6_3 = 46

IS_WORLD = False
IS_CUSTOM_MODEL = True
INVERT_UVS = True

def f32_to_f16(float32):
    F16_EXPONENT_BITS = 0x1F
    F16_EXPONENT_SHIFT = 10
    F16_EXPONENT_BIAS = 15
    F16_MANTISSA_BITS = 0x3ff
    F16_MANTISSA_SHIFT =  (23 - F16_EXPONENT_SHIFT)
    F16_MAX_EXPONENT =  (F16_EXPONENT_BITS << F16_EXPONENT_SHIFT)
    a = struct.pack('>f',float32)
    b = binascii.hexlify(a)
    f32 = int(b,16)
    f16 = 0
    sign = (f32 >> 16) & 0x8000
    exponent = ((f32 >> 23) & 0xff) - 127
    mantissa = f32 & 0x007fffff
    if exponent == 128:
        f16 = sign | F16_MAX_EXPONENT
        if mantissa:
            f16 |= (mantissa & F16_MANTISSA_BITS)
    elif exponent > 15:
        f16 = sign | F16_MAX_EXPONENT
    elif exponent > -15:
        exponent += F16_EXPONENT_BIAS
        mantissa >>= F16_MANTISSA_SHIFT
        f16 = sign | exponent << F16_EXPONENT_SHIFT | mantissa
    else:
        f16 = sign
    return f16

@dataclass
class CompiledMesh:
    positions: list
    boneIndices: list
    blendWeights: list
    normals: list
    uv1s: list
    tangents: Any
    binormals: Any
    hasUvs: bool

    remapData: list[int]
    indices: list[int]

def compact_vertices(indices, positions, uv1s=None, boneIndices=None, blendWeights=None, normals=None, tangents=None, binormals=None):
    remap = {}
    next_id = 0
    for old in indices:
        if old not in remap:
            remap[old] = next_id
            next_id += 1

    def compact_array(arr):
        if arr is None:
            return None
        out = [arr[old] for old in remap.keys()]
        return out
    
    new_positions = compact_array(positions)
    new_uv1s = compact_array(uv1s)
    new_boneIndices = compact_array(boneIndices)
    new_blendWeights = compact_array(blendWeights)
    new_normals = compact_array(normals)
    new_tangents = compact_array(tangents)
    new_binormals = compact_array(binormals)

    new_indices = [remap[old] for old in indices]

    return (new_indices, new_positions, new_uv1s, new_boneIndices,
            new_blendWeights, new_normals, new_tangents, new_binormals)

def compile_mesh(meshName: str, boneNames: list[str], needsTangentBinormal: bool) -> CompiledMesh:
    object = bpy.data.objects[meshName]
    mesh = object.data
    print("Exporting Mesh: " + mesh.name)

    uvLayers = {ul.name: ul for ul in mesh.uv_layers}
    hasUvs = len(uvLayers) > 0
    if hasUvs:
        uvLayerName = mesh.uv_layers.active.name
        # Needed so we don't accidentally dump zeroes for Tangent and Binormal.
        if needsTangentBinormal:
            mesh.calc_tangents(uvmap=uvLayerName)
    
    # NOTE: we will construct per-loop arrays to preserve different UVs on the same vertex
    if not IS_WORLD:
        remapData = set()
        for vi, v in enumerate(mesh.vertices):
            for group in v.groups:
                boneIndex = boneNames.index(object.vertex_groups[group.group].name)
                if group.weight != 0.0:
                    remapData.add(boneIndex)
        remapData = list(remapData)

        # per-vertex positions still used for fallback but we'll use loop_positions for export
        for vi, v in enumerate(mesh.vertices):
            pass
    else:
        remapData = None

    # --- NEW: build per-loop attribute arrays ---
    loop_positions = []
    loop_uv1s = []
    loop_normals = []
    loop_tangents = []
    loop_binormals = []
    loop_boneIndices = []
    loop_blendWeights = []

    for li, loop in enumerate(mesh.loops):
        vi = loop.vertex_index
        v = mesh.vertices[vi]

        # Position (copied from vertex)
        loop_positions.append((v.co.x, v.co.y, v.co.z))

        # Normal per loop
        n = loop.normal
        loop_normals.append((n.x, n.y, n.z))

        # UV per loop
        if hasUvs:
            uv = mesh.uv_layers[uvLayerName].data[li].uv
            loop_uv1s.append((uv.x, 1.0 - uv.y) if INVERT_UVS else (uv.x, uv.y))
        else:
            loop_uv1s.append((0.0, 0.0))

        # Bone data per loop (derived from vertex groups on the vertex)
        if not IS_WORLD:
            bi = [0, 0, 0, 0]
            bw = [0.0, 0.0, 0.0, 0.0]
            for i, group in enumerate(v.groups):
                boneIndex = boneNames.index(object.vertex_groups[group.group].name)
                bi[i] = remapData.index(boneIndex)
                bw[i] = group.weight
            loop_boneIndices.append(bi)
            loop_blendWeights.append(bw)

        # Tangent / binormal per loop
        if needsTangentBinormal:
            t = loop.tangent
            b = t.cross(n)
            loop_tangents.append((t.x, t.y, t.z))
            loop_binormals.append((b.x, b.y, b.z))
        else:
            loop_tangents.append(None)
            loop_binormals.append(None)

    # Build indices using loop indices (so they reference per-loop attributes)
    loop_indices = []
    for tri in mesh.loop_triangles:
        loop_indices.extend(tri.loops)

    # Compact vertices using per-loop arrays so vertices are duplicated when their loop attributes differ
    if needsTangentBinormal:
        indices, positions, uv1s, boneIndices, blendWeights, normals, tangents, binormals = compact_vertices(
            loop_indices,
            loop_positions,
            loop_uv1s,
            loop_boneIndices if not IS_WORLD else None,
            loop_blendWeights if not IS_WORLD else None,
            loop_normals,
            loop_tangents if needsTangentBinormal else None,
            loop_binormals if needsTangentBinormal else None
        )
        return CompiledMesh(positions, boneIndices, blendWeights, normals, uv1s, tangents, binormals, hasUvs, remapData, indices)
    else:
        indices, positions, uv1s, boneIndices, blendWeights, normals, _, _ = compact_vertices(
            loop_indices,
            loop_positions,
            loop_uv1s,
            loop_boneIndices if not IS_WORLD else None,
            loop_blendWeights if not IS_WORLD else None,
            loop_normals,
            None,
            None
        )
        return CompiledMesh(positions, boneIndices, blendWeights, normals, uv1s, None, None, hasUvs, remapData, indices)

def finalize_mesh(mesh: CompiledMesh, needsTangentBinormal: bool) -> tuple[bytes, bytes, bytes]:
    vStream0 = b""
    vStream1 = b""
    iStream = b""
    for i in range(len(mesh.positions)):
        vStream0 += struct.pack('<' + FLOAT_3, *mesh.positions[i])
        if not IS_WORLD:
            vStream0 += struct.pack('<' + U8_4, *mesh.boneIndices[i])
            vStream0 += struct.pack('<' + U8_4_NORMALIZED, int(mesh.blendWeights[i][0] * 255), int(mesh.blendWeights[i][1] * 255), int(mesh.blendWeights[i][2] * 255), int(mesh.blendWeights[i][3] * 255))
        if mesh.hasUvs:
            vStream0 += struct.pack('<' + FLOAT16_2, f32_to_f16(mesh.uv1s[i][0]), f32_to_f16(mesh.uv1s[i][1]))
        if needsTangentBinormal:
            vStream1 += struct.pack('<' + FLOAT16_4, f32_to_f16(mesh.tangents[i][0]), f32_to_f16(mesh.tangents[i][1]), f32_to_f16(mesh.tangents[i][2]), 0)
        vStream1 += struct.pack('<' + FLOAT16_4, f32_to_f16(mesh.normals[i][0]), f32_to_f16(mesh.normals[i][1]), f32_to_f16(mesh.normals[i][2]), 0)
        if needsTangentBinormal:
            vStream1 += struct.pack('<' + FLOAT16_4, f32_to_f16(mesh.binormals[i][0]), f32_to_f16(mesh.binormals[i][1]), f32_to_f16(mesh.binormals[i][2]), 0)
    
    for i in range(len(mesh.indices)):
        iStream += struct.pack('<H', mesh.indices[i])
    
    return vStream0, vStream1, iStream

SCENE_NAME = "ripclutch"
SOURCE_OCT_JSON = "C:\\Projects\\OctaneEngine\\Cars 2 Arcade\\carscade\\assets\\dlc\\car_0008\\frontend\\characters\\francesco_hotrod\\francesco_hotrod\\characters\\francesco_hotrod\\francesco_hotrod.oct.json"
DESTINATION_FOLDER = "C:\\Projects\\OctaneEngine\\BlenderIO\\destination\\"

with open(SOURCE_OCT_JSON) as _tup:
    inTup = json.load(_tup)

# Collects all mesh names into `meshNames`.
meshNames = []
for mesh in bpy.data.objects:
    if bpy.data.objects[mesh.name].type == 'MESH':
        meshNames.append(mesh.name)

# Collect all bone names into `boneNames`.
boneNames = []
if IS_CUSTOM_MODEL:
    boneNames = [
        "Main",
        "Body",
        "FrontEnd",
        "CheekLowerRight",
        "CheekUpperRight",
        "CheekLowerLeft",
        "CheekUpperLeft",
        "LipLowerCornerRight",
        "LipUpperCornerRight",
        "LipLowerMidRight",
        "LipLowerMidLeft",
        "LipLowerCornerLeft",
        "LipUpperCornerLeft",
        "LipUpperMidRight",
        "LipUpperMidLeft",
        "LipLowerCenter",
        "LipUpperCenter",
        "TeethUpper",
        "Jaw",
        "TeethLower",
        "TongueRoot",
        "Tongue1",
        "Tongue2",
        "BackEnd",
        "BrowCornerRight",
        "BrowMidRight",
        "BrowCenter",
        "BrowMidLeft",
        "BrowCornerLeft",
        "TireFLeft_centerJoint",
        "TireFLeft_SpinJoint",
        "TireFLeft_SuspensionJoint",
        "TireFRight_centerJoint",
        "TireFRight_SpinJoint",
        "TireFRight_SuspensionJoint",
        "TireBLeft_centerJoint",
        "TireBLeft_SpinJoint",
        "TireBLeft_SuspensionJoint",
        "TireBRight_centerJoint",
        "TireBRight_SpinJoint",
        "TireBRight_SuspensionJoint"
    ]  
else:
    _boneIdToNameDict = {}
    for node in inTup["SceneTreeNodePool"]:
        if inTup["SceneTreeNodePool"][node]["Type"] == "Bone":
            _boneIdToNameDict[inTup["SceneTreeNodePool"][node]["BoneID"]] = inTup["SceneTreeNodePool"][node]["NodeName"]
    boneNames = [_boneIdToNameDict[i] for i in range(len(_boneIdToNameDict))]
    del _boneIdToNameDict

vBuf = b""
iBuf = b""
outputTup = {}
outputTup["VertexStreamPool"] = {}
outputTup["IndexStreamPool"] = {}
if not IS_WORLD:
    outputTup["SceneComponentPool"] = {}
outputTup["SceneTreeNodePool"] = {}

for i in range(len(meshNames)):
    needsTangentBinormal = "_Tires" in meshNames[i] # FIXME: This is a hack!

    sceneTreeNodeKey = "UnknownMesh#" + str(i)
    if not IS_CUSTOM_MODEL:
        for node in inTup["SceneTreeNodePool"]:
            if "SubGeometry" in inTup["SceneTreeNodePool"][node]["Type"]:
                if inTup["SceneTreeNodePool"][node]["NodeName"] == meshNames[i]:
                    sceneTreeNodeKey = node

    compiled = compile_mesh(meshNames[i], boneNames, needsTangentBinormal)
    vStream0, vStream1, iStream = finalize_mesh(compiled, needsTangentBinormal)
    if IS_WORLD:
        if compiled.hasUvs:
            vStream0Meta = {
                "Length": len(compiled.positions),
                "Width": 16,
                "ExtraStride": 0,
                "VertexBufferReference": 0,
                "VertexBufferOffset": len(vBuf),
                "Elements": {
                    "Element#0": { "Type": 2, "Name": "Position", "Offset": 0 },
                    "Element#1": { "Type": 44, "Name": "Uv1", "Offset": 12 }
                }
            }
        else:
            vStream0Meta = {
                "Length": len(compiled.positions),
                "Width": 12,
                "ExtraStride": 0,
                "VertexBufferReference": 0,
                "VertexBufferOffset": len(vBuf),
                "Elements": {
                    "Element#0": { "Type": 2, "Name": "Position", "Offset": 0 },
                }
            }
    else:        
        vStream0Meta = {
            "Length": len(compiled.positions),
            "Width": 24,
            "ExtraStride": 0,
            "VertexBufferReference": 0,
            "VertexBufferOffset": len(vBuf),
            "Elements": {
                "Element#0": { "Type": 2, "Name": "Position", "Offset": 0 },
                "Element#1": { "Type": 24, "Name": "BoneIndices", "Offset": 12 },
                "Element#2": { "Type": 25, "Name": "BlendWeights", "Offset": 16 },
                "Element#3": { "Type": 44, "Name": "Uv1", "Offset": 20 }
            }
        }
    vBuf += vStream0
    if needsTangentBinormal:
        vStream1Meta = {
            "Length": len(compiled.positions),
            "Width": 24,
            "ExtraStride": 0,
            "VertexBufferReference": 0,
            "VertexBufferOffset": len(vBuf),
            "Elements": {
                "Element#0": { "Type": 45, "Name": "Tangent", "Offset": 0 },
                "Element#1": { "Type": 45, "Name": "Normal", "Offset": 8 },
                "Element#2": { "Type": 45, "Name": "Binormal", "Offset": 16 }
            }
        }
    else:
        vStream1Meta = {
            "Length": len(compiled.positions),
            "Width": 8,
            "ExtraStride": 0,
            "VertexBufferReference": 0,
            "VertexBufferOffset": len(vBuf),
            "Elements": {
                "Element#0": { "Type": 45, "Name": "Normal", "Offset": 0 }
            }
        }
    vBuf += vStream1
    iStreamMeta = {
        "Length": len(compiled.indices),
        "IndexBufferReference": 0,
        "IndexBufferOffset": len(iBuf),
        "IndexStreamPrimitive": "TRIANGLES",
        "IndexStreamElement": ""
    }
    iBuf += iStream
    if not IS_WORLD:
        sceneComponentMeta = {
            "Type": "SkinRemap",
            "RemapData": compiled.remapData
        }
    
    if not IS_WORLD:
        sceneTreeNodeMeta = {
            "NodeName": meshNames[i],
            "VertexStreamReferences": [i * 2, i * 2 + 1],
            "IndexStreamReference": i,
            "SceneComponentReferences": [i + 1]
        }
    else:
        sceneTreeNodeMeta = {
            "NodeName": meshNames[i],
            "VertexStreamReferences": [i * 2, i * 2 + 1],
            "IndexStreamReference": i
        }

    outputTup["VertexStreamPool"]["VertexStream#" + str(i * 2)] = vStream0Meta
    outputTup["VertexStreamPool"]["VertexStream#" + str(i * 2 + 1)] = vStream1Meta
    outputTup["IndexStreamPool"]["IndexStream#" + str(i)] = iStreamMeta
    outputTup["SceneTreeNodePool"][sceneTreeNodeKey] = sceneTreeNodeMeta
    if not IS_WORLD:
        outputTup["SceneComponentPool"]["SceneComponent#" + str(i + 1)] = sceneComponentMeta

outputTup["VertexBufferPool"] = { "VertexBuffer#0": { "Name": "Static", "Flags": 73, "Size": len(vBuf), "HeapLoc": 0, "FileName": SCENE_NAME + "_0.vbuf" } }
outputTup["IndexBufferPool"] = { "IndexBuffer#0": { "Width": 2, "Name": "Static", "Flags": 91, "Size": len(iBuf), "FileName": SCENE_NAME + "_0.ibuf" } }

with open(DESTINATION_FOLDER + SCENE_NAME + ".patch.json", "w") as tup:
    json.dump(outputTup, tup, indent=2)

with open(DESTINATION_FOLDER + SCENE_NAME + "_0.ibuf", "wb") as outIBuf:
    outIBuf.write(iBuf)

with open(DESTINATION_FOLDER + SCENE_NAME + "_0.vbuf", "wb") as outVBuf:
    outVBuf.write(vBuf)
