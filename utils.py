import torch


def compute_best_buddies(frame1, frame2):
    """
        Input: tensor1 (w1,h1,c) and tensor2 (w2,h2,c)
        Output: 
            - a (w1,h1,2) tensor containing the best buddy of each entry. If a best buddy was not found, the entry is -1.
            - n: a scalar tensor representing the number of best buddies found
            - delta_pos: the sum of displacements of best buddies
    """
    w1, h1, c = frame1.shape
    w2, h2, c = frame2.shape
    frame1 = frame1.view(w1 * h1, c)
    frame2 = frame2.view(w2 * h2, c)

    # Compute the distance matrix
    dist = torch.cdist(frame1, frame2, p=2)
    
    # Find the best buddy for each entry
    nearest_neighbors_of_frame1 = torch.argmin(dist, dim=1)
    nearest_neighbors_of_frame2 = torch.argmin(dist, dim=0)

    # two entries are best buddies if they are each other's nearest neighbors
    best_buddies = torch.zeros(w1 * h1, 2, dtype=torch.int, device=frame1.device)
    n = torch.tensor(0.0, device=frame1.device)
    delta_pos = torch.tensor(0.0, device=frame1.device)

    for i in range(w1 * h1):
        if i == nearest_neighbors_of_frame2[nearest_neighbors_of_frame1[i]]:
            best_buddies[i] = torch.tensor([nearest_neighbors_of_frame1[i].item()//h2, nearest_neighbors_of_frame1[i].item()%h2], device=frame1.device)
            n += 1.0
            delta_pos += dist[i][nearest_neighbors_of_frame1[i]]
        else:
            best_buddies[i] = torch.tensor(-1.0, device=frame1.device)

    return best_buddies.view(w1, h1, 2), n, delta_pos


def compute_best_buddies_continuity_loss(frame1, frame2, frame3):
    """
        Input: tensor1 (w1,h1,c), tensor2 (w2,h2,c), tensor3 (w3,h3,c)
        Output: 
            - loss: a scalar tensor representing the loss
            - n: a scalar tensor representing the number of best buddies found between all 3 frames
    """
    w2, h2, _ = frame2.shape
    best_buddies21, _, _ = compute_best_buddies(frame2, frame1)
    best_buddies23, _, _ = compute_best_buddies(frame2, frame3)

    loss = torch.tensor(0.0, device=frame1.device)
    n = torch.tensor(0.0, device=frame1.device)
    for i in range(w2):
        for j in range(h2):
            if best_buddies21[i][j][0] != -1 and best_buddies23[i][j][0] != -1:
                # check if average of frame1[i][j] and frame3[best_buddies12[i][j]] is close to frame2[best_buddies13[i][j]]
                expected_frame2 = (frame1[best_buddies21[i][j][0]][best_buddies21[i][j][1]] + frame3[best_buddies23[i][j][0]][best_buddies23[i][j][1]])/2
                loss += torch.dist(expected_frame2, frame2[i][j])
                n += 1.0
    return loss, n

if __name__ == "__main__":
    torch.manual_seed(0)
    # Test the function
    frame1 = torch.rand(5, 2, 2)
    frame2 = torch.rand(4, 3, 2)
    frame3 = torch.rand(3, 3, 2)
    print(compute_best_buddies_continuity_loss(frame1, frame2, frame3))
