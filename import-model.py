import struct, json
from typing import Optional
import numpy as np
import bpy
from mathutils import *
from dataclasses import dataclass

FLOAT_2 = "ff" # 1
FLOAT_3 = "fff" # 2
FLOAT_4 = "ffff" # 3
U8_4 = 'BBBB' # 24
U8_4_NORMALIZED = 'BBBB' # 25
S10_10_10_NORMALIZED = 'I' # 31
S11_11_10_NORMALIZED = 'I' # 39
FLOAT16_2 = 'HH' # 44
FLOAT16_4 = 'HHHH' # 45
S1_6_3 = 'bbb' # 46

ID_FLOAT_2 = 1
ID_FLOAT_3 = 2
ID_FLOAT_4 = 3
ID_U8_4 = 24
ID_U8_4_NORMALIZED = 25
ID_S10_10_10_NORMALIZED = 31
ID_S11_11_10_NORMALIZED = 39
ID_FLOAT16_2 = 44
ID_FLOAT16_4 = 45
ID_S1_6_3 = 46

IS_WORLD = False

def f16_to_f32(float16: int) -> float:
    s = int((float16 >> 15) & 0x00000001)
    e = int((float16 >> 10) & 0x0000001f)
    f = int(float16 & 0x000003ff)
    if e == 0:
        if f == 0:
            ret = int(s << 31)
            return struct.unpack(">f", ret.to_bytes(4, "big"))[0]
        else:
            while not (f & 0x00000400):
                f = f << 1
                e -= 1
            e += 1
            f &= ~0x00000400
    elif e == 31:
        if f == 0:
            ret = int((s << 31) | 0x7f800000)
            return struct.unpack(">f", ret.to_bytes(4, "big"))[0]
        else:
            ret = int((s << 31) | 0x7f800000 | (f << 13))
            return struct.unpack(">f", ret.to_bytes(4, "big"))[0]
    e = e + (127 -15)
    f = f << 13
    ret = int((s << 31) | (e << 23) | f)
    return struct.unpack(">f", ret.to_bytes(4, "big"))[0]

def sign_extend(value: int, bits: int) -> int:
    sign_bit = 1 << (bits - 1)
    return (value & (sign_bit - 1)) - (value & sign_bit)

def to_floats_s10(int_value: int) -> tuple[float, float, float]:
    raw1 = (int_value >> 0) & 0x3FF
    raw2 = (int_value >> 10) & 0x3FF
    raw3 = (int_value >> 20) & 0x3FF
    sv1 = sign_extend(raw1, 10) / 511
    sv2 = sign_extend(raw2, 10) / 511
    sv3 = sign_extend(raw3, 10) / 511
    return sv1, sv2, sv3

def to_floats_s11_11_10(int_value: int) -> tuple[float, float, float]:
    raw1 = (int_value >>  0) & 0x7FF
    raw2 = (int_value >> 11) & 0x7FF
    raw3 = (int_value >> 22) & 0x3FF
    sv1 = sign_extend(raw1, 11) / 1023
    sv2 = sign_extend(raw2, 11) / 1023
    sv3 = sign_extend(raw3, 10) / 511
    return sv1, sv2, sv3

def mat4_from_list(flat16):
    assert len(flat16) == 16
    rows = (flat16[0:4], flat16[4:8], flat16[8:12], flat16[12:16])
    m = Matrix(rows)
    return m.transposed() # We need to transpose, since translation needs to be in column 3 in Blender.

@dataclass
class IndexStreamMeta:
    length: int
    iBufIndex: int
    iBufOffset: int

def read_vertex_streams(tup: dict, vstreams: list[int], vBufMap: dict, remapData: Optional[list[int]]) -> dict:
    elementMapFinal = {}
    for streamIdx in vstreams:
        stream = tup["VertexStreamPool"]["VertexStream#" + str(streamIdx)]
        for elementTag in stream["Elements"]:
            elementName = stream["Elements"][elementTag]["Name"]
            elementMapFinal[elementName] = []

    for streamIdx in vstreams:
        stream = tup["VertexStreamPool"]["VertexStream#" + str(streamIdx)]
        vbuf = open(vBufMap[stream["VertexBufferReference"]], "rb")
        for vertexIndex in range(stream["Length"]):
            for elementTag in stream["Elements"]:
                element = stream["Elements"][elementTag]
                vbuf.seek(stream["VertexBufferOffset"] + vertexIndex * stream["Width"] + element["Offset"])
                ty = element["Type"]
                if ty == ID_FLOAT_2:
                    nibble = struct.unpack('<' + FLOAT_2, vbuf.read(struct.calcsize(FLOAT_2)))
                elif ty == ID_FLOAT_3:
                    nibble = struct.unpack('<' + FLOAT_3, vbuf.read(struct.calcsize(FLOAT_3)))
                elif ty == ID_FLOAT_4:
                    nibble = struct.unpack('<' + FLOAT_4, vbuf.read(struct.calcsize(FLOAT_4)))
                    if "Normal" == element["Name"]:
                        nibble = nibble[0], nibble[1], nibble[2]
                elif ty == ID_U8_4:
                    nibble = struct.unpack('<' + U8_4, vbuf.read(struct.calcsize(U8_4)))
                    # If we encounter BoneIndices, we need to remap them via remapData.
                    if "BoneIndices" == element["Name"]:
                        if IS_WORLD:
                            raise ValueError
                        normalizedNibble = list(nibble)
                        for i in range(len(nibble)):
                            if nibble[i] >= len(remapData):
                                normalizedNibble[i] = 0
                        nibble = remapData[normalizedNibble[0]], remapData[normalizedNibble[1]], remapData[normalizedNibble[2]], remapData[normalizedNibble[3]]
                elif ty == ID_U8_4_NORMALIZED:
                    nibble = struct.unpack('<' + U8_4_NORMALIZED, vbuf.read(struct.calcsize(U8_4_NORMALIZED)))
                elif ty == ID_S10_10_10_NORMALIZED:
                    nibble = to_floats_s10(struct.unpack('<' + S10_10_10_NORMALIZED, vbuf.read(struct.calcsize(S10_10_10_NORMALIZED)))[0])
                elif ty == ID_S11_11_10_NORMALIZED:
                    nibble = to_floats_s11_11_10(struct.unpack('<' + S11_11_10_NORMALIZED, vbuf.read(struct.calcsize(S11_11_10_NORMALIZED)))[0])
                elif ty == ID_FLOAT16_2:
                    nibble = struct.unpack('<' + FLOAT16_2, vbuf.read(struct.calcsize(FLOAT16_2)))
                    nibble = f16_to_f32(nibble[0]), f16_to_f32(nibble[1])
                elif ty == ID_FLOAT16_4:
                    nibble = struct.unpack('<' + FLOAT16_4, vbuf.read(struct.calcsize(FLOAT16_4)))
                    nibble = f16_to_f32(nibble[0]), f16_to_f32(nibble[1]), f16_to_f32(nibble[2]), f16_to_f32(nibble[3])
                    if "Normal" == element["Name"]:
                        nibble = nibble[0], nibble[1], nibble[2]
                else:
                    raise ValueError(f"Unhandled Type: {ty}! Aborting...")
                elementMapFinal[element["Name"]].append(nibble)
        vbuf.close()
    return elementMapFinal

SCENE = "mater"
SCENE_FOLDER = "C:\\Projects\\OctaneEngine\\Cars2\\BlenderIO\\source\\"

with open(SCENE_FOLDER + SCENE + ".oct.json") as tmp:
    tup = json.load(tmp)

vBufMap = {}

for buffer in tup["VertexBufferPool"]:
    if tup["VertexBufferPool"][buffer]["Name"] == "Static" and "FileName" in tup["VertexBufferPool"][buffer]:
        bufferIndex = int(buffer.split("#")[1])
        vBufMap[bufferIndex] = SCENE_FOLDER + tup["VertexBufferPool"][buffer]["FileName"]

iBufMap = {}

for buffer in tup["IndexBufferPool"]:
    if tup["IndexBufferPool"][buffer]["Name"] == "Static" and "FileName" in tup["IndexBufferPool"][buffer]:
        bufferIndex = int(buffer.split("#")[1])
        iBufMap[bufferIndex] = SCENE_FOLDER + tup["IndexBufferPool"][buffer]["FileName"]

istreams = []

for istream in tup["IndexStreamPool"]:
    istreams.append(IndexStreamMeta(tup["IndexStreamPool"][istream]["Length"], tup["IndexStreamPool"][istream]["IndexBufferReference"], tup["IndexStreamPool"][istream]["IndexBufferOffset"]))

boneIdToName = []
_boneIdToNameDict = {}

for node in tup["SceneTreeNodePool"]:
    if tup["SceneTreeNodePool"][node]["Type"] == "Bone":
        _boneIdToNameDict[tup["SceneTreeNodePool"][node]["BoneID"]] = tup["SceneTreeNodePool"][node]["NodeName"]

boneIdToName = [_boneIdToNameDict[i] for i in range(len(_boneIdToNameDict))]
del _boneIdToNameDict

for node in tup["SceneTreeNodePool"]:
    if "SubGeometry" in tup["SceneTreeNodePool"][node]["Type"]:
        # Grab the mesh name
        meshName = tup["SceneTreeNodePool"][node]["NodeName"]
        
        # Get the vertex and index stream indices
        vstreamIndices = tup["SceneTreeNodePool"][node]["VertexStreamReferences"]
        istreamIndex = tup["SceneTreeNodePool"][node]["IndexStreamReference"]
        
        # Get RemapData
        if not IS_WORLD:
            sceneComponentRefs = tup["SceneTreeNodePool"][node]["SceneComponentReferences"]
            assert(len(sceneComponentRefs) == 1)
            skinRemap = tup["SceneComponentPool"]["SceneComponent#" + str(sceneComponentRefs[0])]
            assert(skinRemap["Type"] == "SkinRemap")
            remapData = skinRemap["RemapData"]
        else:
            remapData = None
        print(remapData)
        
        mesh = bpy.data.meshes.new(meshName)
        obj = bpy.data.objects.new(meshName, mesh)
        bpy.context.collection.objects.link(obj)
        
        istream = istreams[istreamIndex]
        indexData = np.fromfile(iBufMap[istream.iBufIndex], dtype='<u2', count=istream.length, offset=istream.iBufOffset)

        vstreams = read_vertex_streams(tup, vstreamIndices, vBufMap, remapData)

        faces = [tuple(indexData[i:i+3]) for i in range(0, len(indexData), 3)]
        mesh.from_pydata([Vector(pos) for pos in vstreams["Position"]], [], faces)

        if 'Normal' in vstreams:
            normals = vstreams['Normal']
            mesh.normals_split_custom_set_from_vertices(normals)
        
        for element in vstreams:
            if element.startswith('Uv'):
                uv_layer = mesh.uv_layers.new(name=element)
                uvs = vstreams[element]
                for li, loop in enumerate(mesh.loops):
                    v_idx = loop.vertex_index
                    uv_layer.data[li].uv = uvs[v_idx]
        
        if 'BoneIndices' in vstreams and 'BlendWeights' in vstreams:
            boneIndices = vstreams['BoneIndices']
            blendWeights = vstreams['BlendWeights']
            # Add all necessary vertex groups
            for bone in remapData:
                if boneIdToName[bone] not in obj.vertex_groups:
                    obj.vertex_groups.new(name=boneIdToName[bone])
            # Assign indices and weights through the vertex groups 
            for vertexIndex in range(len(boneIndices)):
                bi = boneIndices[vertexIndex]
                bw = blendWeights[vertexIndex]

                for i in range(4):
                    if bw[i] <= 0.0:
                        continue
                    vertexGroup = obj.vertex_groups[boneIdToName[bi[i]]]
                    vertexGroup.add([vertexIndex], bw[i] / 255, 'REPLACE')

def find_scene_tree_node(tup, nodeName):
    for node in tup["SceneTreeNodePool"]:
        if "NodeName" in tup["SceneTreeNodePool"][node]:
            if tup["SceneTreeNodePool"][node]["NodeName"] == nodeName:
                return tup["SceneTreeNodePool"][node]
    return {}

# Armature time!
localToParentMats = {}
for node in tup["SceneTreeNodePool"]:
    if tup["SceneTreeNodePool"][node]["Type"] == "Bone":
        idx = tup["SceneTreeNodePool"][node]["BoneID"]
        localToParentMats[idx] = mat4_from_list(tup["SceneTreeNodePool"][node]["LocalToParentMatrix"])

worldMats = {}

def compute_world(boneId):
    if boneId in worldMats:
        return worldMats[boneId]
    parentSceneTreeNodeIndex = find_scene_tree_node(tup, boneIdToName[boneId])["ParentNodeReferences"][0]
    local = localToParentMats[boneId]
    if boneIdToName[boneId] != 'Main':
        parentId = tup["SceneTreeNodePool"]["Node#" + str(parentSceneTreeNodeIndex)]["BoneID"]
        parentWorld = compute_world(parentId)
        w = parentWorld @ local
    else:
        # We're handling a root, bone, so its world matrix is the same as the local -> parent matrix.
        w = local.copy()
    worldMats[boneId] = w
    return w

for i in range(len(boneIdToName)):
    compute_world(i)

armatureData = bpy.data.armatures.new(SCENE)
armatureObj = bpy.data.objects.new(SCENE, armatureData)
bpy.context.collection.objects.link(armatureObj)
bpy.context.view_layer.objects.active = armatureObj
armatureObj.select_set(True)

# Here, we switch to Edit mode to manipulate edit bones.
bpy.ops.object.mode_set(mode='EDIT')
editBones = armatureData.edit_bones

for i in range(len(boneIdToName)):
    eb = editBones.new(boneIdToName[i])
    eb.head = Vector((0.0,0.0,0.0))
    eb.tail = Vector((0.0,0.1,0.0))

for i in range(len(boneIdToName)):
    if boneIdToName[i] != "Main":
        parentSceneTreeNodeIndex = find_scene_tree_node(tup, boneIdToName[i])["ParentNodeReferences"][0]
        parentSceneTreeNode = tup["SceneTreeNodePool"]["Node#" + str(parentSceneTreeNodeIndex)]
        parentName = parentSceneTreeNode["NodeName"]
        editBones[boneIdToName[i]].parent = editBones[parentName]

for i in range(len(boneIdToName)):
    name = boneIdToName[i]
    eb = editBones[name]
    world = worldMats[i]
    # edit_bone.matrix expects a matrix in armature space (4x4)
    # Set the matrix directly:
    eb.matrix = world

    # Set head/tail if PivotPoint/BoneEndPoint provided (preferred for visualization)
    pivot = Vector(find_scene_tree_node(tup, name)["PivotPoint"])
    # pivot = Vector((pivot.x, pivot.y, pivot.z, 1.0))
    eb.head = pivot

    if "BoneEndPoint" in find_scene_tree_node(tup, name):
        endp = Vector(find_scene_tree_node(tup, name)["BoneEndPoint"])
        endp = Vector((endp.x, endp.y, endp.z))
        # ensure tail != head (Blender fails on zero-length)
        if (endp - eb.head).length < 1e-6:
            eb.tail = eb.head + Vector((0.0, 0.05, 0.0))
        else:
            eb.tail = endp
    else:
        # fallback: set a small tail along bone Y (or take head + local y from world matrix)
        # Extract local Y axis of the bone in world to make a sensible tail
        localY = world.to_3x3() @ Vector((0.0, 1.0, 0.0))
        eb.tail = eb.head + localY.normalized() * 0.1

    # If head exactly equals parent's tail, set as connected
    if eb.parent:
        if (eb.head - eb.parent.tail).length < 1e-6:
            eb.use_connect = True

# Leave Edit mode.
bpy.ops.object.mode_set(mode='OBJECT')

# Parent the Armature to every mesh.
allMeshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
for mesh in allMeshes:
    mod = mesh.modifiers.new(name="Armature", type='ARMATURE')
    mod.object = armatureObj
    mesh.parent = armatureObj
    mesh.parent_type = 'ARMATURE'