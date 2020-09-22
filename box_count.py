box_top_s = 150 #���|����(�W�L�N���Ѧ����|)
box_space = 2 #�ϰ�Ӽ�
box_line = [850, 1400] #�ϰ���νu�A�ѥ���k�Ƨ�(X�b)
box_bottom = 650 #�ϰ쩳�u(Y�b)
def box_count(detections, box_top_s, box_space, box_line, box_bottom):
  box_num = np.zeros((box_space))
  y_top_list = [[] for i in range(box_space)]
  y_top_box_list = [[] for i in range(box_space)]
  for detection in detections:
	label = detection[0]
	bounds = detection[2]
	box_height = int(bounds[3])
	box_width = int(bounds[2])
    # �p�� Box �y��
	x_left = int(bounds[0] - bounds[2]/2)
	y_top = int(bounds[1] - bounds[3]/2)
	#�L�o�ؼаϰ�H�~����l
    if x_left + box_width > box_line[box_space - 1] or y_top > box_bottom: # or confidence < 0.9
      continue
	#�ھڰϰ�ƶq�A��W�x�sbox_top�Mbox��y_top
    for i in range(box_space):
      if label == "box_top":
        if x_left + box_width <= box_line[i]:
          y_top_list[i].append(y_top)
          box_num[i] = box_num[i] + 1
          break
      elif label == "box":
        if x_left + box_width <= box_line[i]:
          y_top_box_list[i].append(y_top)
          break
  #list to numpy array
  for i in range(box_space):
    y_top_list[i] = np.array(y_top_list[i])
    y_top_box_list[i] = np.array(y_top_box_list[i])
  #�B�z��l���|���D
  for i in range(box_space):
    #�P�_�ϰ줺�O�_����l
    if y_top_list[i].size != 0:
      diff_num = y_top_list[i][(y_top_list[i] - y_top_list[i].min()) > box_top_s] #�Ĥ@�h�@����
	  #�p���l�ƶq
	  #�p��y�{�G�N�Ĥ@�h����l�L�o���A�N�ѤU����l�ƶq���H�h���A�A�[�W�Ĥ@�h����l�ƶq
      if diff_num.size != 0:
        box_num[i] = (box_num[i] - diff_num.size) * 2 + diff_num.size
      else:
	  #box_top���b�P�@�h�ɡA�Nbox_top�ƶq���H�h���A�Y�@����ƶq
        if y_top_list[i].size != 0:
          if (y_top_box_list[i].max() - y_top_box_list[i].min()) > box_top_s:
            box_num[i] = box_num[i] * 2
  return box_num