import torch
import torch.nn as nn
import torch.nn.functional as F
from utility_fun import truth_checker

class training_class:
    def __init__(self, model, trainloader, testloader, device, epoch, optimizer,criterion,l1=0,l2=0,scheduler = None):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device
        self.epoch = epoch
        self.optimizer = optimizer
        self.criterion = criterion
        self.l1 = l1
        self.l2 = l2
        self.loss = 0
        self.scheduler = scheduler
        self.metric  = 0

    def fit(self):
        self.model.train()
        train_acc = []
        test_acc = []
        test_loss = []
        for x_epoch in range(self.epoch):
            #running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                # Loss

                self.loss = self.criterion(outputs, labels)
                if self.l1 > 0:
                    l1_loss = 0
                    for param in self.model.parameters():
                        l1_loss += torch.norm(param,1)
                    self.loss += self.l1*l1_loss
                if self.l2 > 0:
                    l2_loss = 0
                    for param in self.model.parameters():
                        l2_loss +=torch.norm(param,2)
                    self.loss += self.l2*l2_loss


                self.loss.backward()
                self.optimizer.step()
                
            ###Where to Predict LEL
            _,train_check = truth_checker(self.model,self.trainloader,self.device,self.criterion) ##FIXTHISSHITLATER
            testloss,test_check =  truth_checker(self.model, self.testloader, self.device,self.criterion)
            train_acc.append(train_check)
            test_acc.append(test_check)
            test_loss.append(testloss)

            if self.scheduler:
                self.scheduler.step(test_loss[-1])

            #train_check = truth_checker(self.model,self.trainloader)
            print('epoch [%d] train accuracy %.3f : test accuracy %.3f' %
                  (x_epoch,train_check, test_check))
        return train_acc,test_acc

    def predict_method(self,testloader,miss_class,correct_class,remode=None):
        self.testloader = testloader
        self.miss_class = miss_class
        self.correct_class = correct_class
        self.remode = remode
        if self.remode == None:
            self.model.eval()
        else:
            self.model = self.remode
            self.model.eval()
        test_loss = 0
        correct = 0
        missclassified_images = []
        correct_images = []
        test_losses = []
        with torch.no_grad():
            for data,target in self.testloader:
                data,target = data.to(self.device),target.to(self.device)
                outputs = self.model(data)
                test_loss +=self.criterion(outputs, target).item()
                pred = outputs.argmax(dim=1, keepdim=True) 
                is_correct = pred.eq(target.view_as(pred))
                if self.miss_class>0:
                    misclassified_inds = (is_correct==0).nonzero()[:,0]
                    for image in misclassified_inds:
                        if len(missclassified_images) == self.miss_class:
                            break
                        missclassified_images.append({"target": target[image].cpu().numpy(),
                                                      "pred": pred[image][0].cpu().numpy(),
                                                      "img": data[image]
                                                      })
                if self.correct_class>0:
                    corret_inds = (is_correct==1).nonzero()[:,0]
                    for image in corret_inds:
                        if len(correct_images)==self.correct_class:
                            break
                        correct_images.append({"target": target[image].cpu().numpy(),
                                                "pred" : pred[image][0].cpu().numpy(),
                                                "img"  : data[image]
                                              })
        correct += is_correct.sum().item()
        test_losses.append(test_loss)
        test_acc = 100 * (correct / len(self.testloader.dataset))
        #test_acc.append(test_acc)
        return test_acc,correct_images,missclassified_images

    def give_model(self):
        return self.model



