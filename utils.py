import torch

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    
def generate_pixel_chain(image_depth):
    labels_map = {0:(-1, 0), 1:(-1, 1), 2:(0, 1), 3:(1, 1), 4:(1, 0), 5:(1, -1), 6:(0, -1), 7:(-1, -1)}
    h, w = image_depth.shape
    pixel_dir = torch.zeros((h, w))-1
    for i in range(h):
      for j in range(w):
        for idx in range(len(labels_map)):
          closest_depth = float('inf')
          dx, dy = labels_map[idx]
          new_i, new_j = i+dx, j+dy
          if new_i >= 0 and new_i < h and new_j >= 0 and new_j < w:
            new_depth = image_depth[i+dx, j+dy]
            if new_depth > image_depth[i, j] and new_depth < closest_depth:
              closest_depth = new_depth
              pixel_dir[i, j] = idx
    return pixel_dir

if __name__=="__main__":
    print(generate_pixel_chain(torch.randn(20, 20)))