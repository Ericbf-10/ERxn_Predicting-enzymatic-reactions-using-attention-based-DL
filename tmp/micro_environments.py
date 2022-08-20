def micro_environment_type1(atom_boxes):
    '''
    Takes atom boxes and returns a sequence of micro environments by deviding the atom box
    into equally sized micro environments
    '''
    carbon_box = atom_boxes[0]
    oxygen_box = atom_boxes[1]
    nitrogen_box = atom_boxes[2]
    sulfur_box = atom_boxes[3]

    grid_x = [x * 5 for x in range(carbon_box.shape[0] // MICRO_BOX_SIZE)]
    grid_y = [y * 5 for y in range(carbon_box.shape[1] // MICRO_BOX_SIZE)]
    grid_z = [z * 5 for z in range(carbon_box.shape[2] // MICRO_BOX_SIZE)]
    micro_env_sequence = []
    for x in grid_x:
        for y in grid_y:
            for z in grid_z:
                cbox = carbon_box[x:x + MICRO_BOX_SIZE, y:y + MICRO_BOX_SIZE, z:z + MICRO_BOX_SIZE]
                obox = oxygen_box[x:x + MICRO_BOX_SIZE, y:y + MICRO_BOX_SIZE, z:z + MICRO_BOX_SIZE]
                nbox = nitrogen_box[x:x + MICRO_BOX_SIZE, y:y + MICRO_BOX_SIZE, z:z + MICRO_BOX_SIZE]
                sbox = sulfur_box[x:x + MICRO_BOX_SIZE, y:y + MICRO_BOX_SIZE, z:z + MICRO_BOX_SIZE]
                micro_env_sequence.append(np.stack([cbox, obox, nbox, sbox]))

    return micro_env_sequence

# devide 3d box into sequence of cubes with dimension MICRO_BOX_SIZE x MICRO_BOX_SIZE x MICRO_BOX_SIZE
micro_env_sequence = micro_environment_type1(atom_boxes)

boxes_with_atoms = 0
for i, box in enumerate(micro_env_sequence):
    C_box = box[0]
    if np.all((C_box == 0)):
        pass
    else:
        boxes_with_atoms += 1
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_aspect('auto')
        ax.voxels(C_box, edgecolor="k")
        plt.title(f'{i}/{len(micro_env_sequence)}')
        plt.show(block=False)
        plt.pause(0.01)
        plt.close()
print(boxes_with_atoms)