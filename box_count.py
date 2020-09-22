box_top_s = 150 #堆疊高度(超過代表桶槽有堆疊)
box_space = 2 #區域個數
box_line = [850, 1400] #區域分割線，由左到右排序(X軸)
box_bottom = 650 #區域底線(Y軸)
def box_count(detections, box_top_s, box_space, box_line, box_bottom):
  box_num = np.zeros((box_space))
  y_top_list = [[] for i in range(box_space)]
  y_top_box_list = [[] for i in range(box_space)]
  for detection in detections:
	label = detection[0]
	bounds = detection[2]
	box_height = int(bounds[3])
	box_width = int(bounds[2])
    # 計算 Box 座標
	x_left = int(bounds[0] - bounds[2]/2)
	y_top = int(bounds[1] - bounds[3]/2)
	#過濾目標區域以外的桶子
    if x_left + box_width > box_line[box_space - 1] or y_top > box_bottom: # or confidence < 0.9
      continue
	#根據區域數量，單獨儲存box_top和box的y_top
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
  #處理桶子堆疊問題
  for i in range(box_space):
    #判斷區域內是否有桶子
    if y_top_list[i].size != 0:
      diff_num = y_top_list[i][(y_top_list[i] - y_top_list[i].min()) > box_top_s] #第一層一噸桶
	  #計算桶子數量
	  #計算流程：將第一層的桶子過濾掉，將剩下的桶子數量乘以層高，再加上第一層的桶子數量
      if diff_num.size != 0:
        box_num[i] = (box_num[i] - diff_num.size) * 2 + diff_num.size
      else:
	  #box_top都在同一層時，將box_top數量乘以層高，即一噸桶數量
        if y_top_list[i].size != 0:
          if (y_top_box_list[i].max() - y_top_box_list[i].min()) > box_top_s:
            box_num[i] = box_num[i] * 2
  return box_num