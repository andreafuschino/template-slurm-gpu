#!/usr/bin/env python

from mining_pairs import get_pairs
import torch


def test_valid_dist(dataloader, model, distNet, loss_fn, device):
  with torch.no_grad():
    model.eval()
    distNet.eval()
    val_loss = 0
    for batch_idx, (data, labels) in enumerate(dataloader):
        data, labels = data.to(device), labels.to(device)
        embeddings = model(data)

        diff_pairs, positive_pairs, negative_pairs, all_pairs_indices = get_pairs(embeddings, labels, distNet, loss_fn, device)
        diff_pairs = diff_pairs.to(device)
        
        dist=distNet(diff_pairs)
        loss=loss_fn(dist, positive_pairs, negative_pairs, all_pairs_indices,labels, embeddings)
        val_loss += loss

    return val_loss
    

def trainDistanceModule(model, distNet, loss_func, device, train_dataloader,valid_dataloader, optimizer, epoch, loss_history_train, loss_history_valid, scheduler):
    model.eval()
    distNet.train()
    total_loss = 0

    #scheduler.step()
    #print("Lr=",get_lr(optimizer))
    for batch_idx, (data, labels)  in enumerate(train_dataloader):
        data, labels = data.to(device), labels.to(device)
        embeddings = model(data)

        #print(labels)
        #print(embeddings)

        #for k in range(0,3):
  
        diff_pairs, positive_pairs, negative_pairs, all_pairs_indices = get_pairs(embeddings, labels, distNet, loss_func, device)
        distNet.train()
        diff_pairs = diff_pairs.to(device)
        
        dist=distNet(diff_pairs)
        loss=loss_func(dist, positive_pairs, negative_pairs, all_pairs_indices,labels, embeddings)
           
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            #scheduler.step(loss)
            #print("Lr=",get_lr(optimizer))
            #loss_history_train.append(loss.item())
            print("Epoch {} Iteration {}: Loss = {}".format(epoch, batch_idx, loss.item()))
            
        if batch_idx==0: break

      
    total_loss /= (batch_idx + 1)
    loss_history_train.append(total_loss)    
    print('Epoch: {}. Train set - Average loss: {:.4f}'.format(epoch, total_loss))
    
    val_loss=test_valid_dist(valid_dataloader, model, distNet, loss_func,device)
    val_loss /= len(valid_dataloader)
    loss_history_valid.append(val_loss)
    print('Epoch: {}. Validation set - Average loss: {:.4f}\n'.format(epoch, val_loss))
    
    #scheduler.step(total_loss)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': distNet.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_train': total_loss,
        'loss_valid': val_loss,
        }, "distNet_random100.pt")



