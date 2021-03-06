import json
import pandas as pd 
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import numpy as np

class Annotator:
    def __init__(self, in_path, out_path):
        self.in_path = in_path
        self.out_path = out_path
        self.obj = self._load_json()
        self.parse_df = self._parsed_annoted_json()
        self.kmeans_map = {}
        
    def _load_json(self):
        with open(self.in_path,'r') as myfile:
            data = myfile.read()
        obj = json.loads(data)
        return obj 
    
    def _parsed_annoted_json(self):
        image_id = []
        category_id = []
        bbox_x = []
        bbox_y = []
        bbox_w = []
        bbox_h = []
        for i in self.obj['annotations']:
            image_id.append(i['image_id'])
            category_id.append(i['category_id'])
            bbox_x.append(i['bbox'][0])
            bbox_y.append(i['bbox'][1])
            bbox_w.append(i['bbox'][2])
            bbox_h.append(i['bbox'][3])
        df = pd.DataFrame(list(zip(image_id,category_id,bbox_x,bbox_y,bbox_w,bbox_h)), columns = ['id', 'category_id', 'bbox_x','bbox_y','bbox_w','bbox_h'])
        df_image = pd.DataFrame(self.obj['images'])
        df_cat = pd.DataFrame(self.obj['categories'])
        df_cat.columns = ['supercategory','category_id','category_name']
        df = pd.merge(df_image[['id','file_name','width','height']], df, on = 'id', how = 'left')
        df = pd.merge(df, df_cat[['category_id','category_name']], on = 'category_id', how = 'left')    
        df['scaled_bbox_x'] = df.apply(lambda x: x['bbox_x']/x['width'], axis = 1)
        df['scaled_bbox_y'] = df.apply(lambda x: x['bbox_y']/x['height'], axis = 1)
        df['scaled_bbox_w'] = df.apply(lambda x: x['bbox_w']/x['width'], axis = 1)
        df['scaled_bbox_h'] = df.apply(lambda x: x['bbox_h']/x['height'], axis = 1)
        df['log_scaled_bbox_w'] = df.apply(lambda x: np.log(x['scaled_bbox_w']), axis = 1)
        df['log_scaled_bbox_h'] = df.apply(lambda x: np.log(x['scaled_bbox_h']), axis = 1)  
        #print(df.head())
        df.to_csv(self.out_path, index = False)
        return (df)   
    
    def show_bboxes(self):
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("Bounding Boxes")
        axs[0].scatter(self.parse_df['scaled_bbox_w'], self.parse_df['scaled_bbox_h'])
        axs[0].set_xlabel('width')
        axs[0].set_ylabel('height')
        axs[1].scatter(self.parse_df['log_scaled_bbox_w'], self.parse_df['log_scaled_bbox_h'])
        axs[1].set_xlabel('log(width)')
        axs[1].set_ylabel('log(height)')
        fig.tight_layout(pad=3.0)
        fig.savefig("bboxes.png")
        plt.show()
        
    def _compute_iou(self,w1, h1, w2, h2):
        intersection = min(w1, w2) * min(h1, h2)
        union = (w1 * h1) + (w2 * h2) - intersection
        iou = intersection/union
        return iou
    
    def create_cluster(self,K):
        sample = self.parse_df.shape[0]
        WCSS = []
        MIOU = []
        #kmeans_map = {}
        for k in range(K):
            kmeans = KMeans(n_clusters=k+1, random_state=0).fit(self.parse_df[['scaled_bbox_w','scaled_bbox_h']])
            wcss = kmeans.inertia_
            cluster_block = kmeans.labels_
            mean_iou = 0
            for i in range(sample):
                which_k = cluster_block[i]
                w_c,h_c = kmeans.cluster_centers_[which_k]
                w_s = self.parse_df['scaled_bbox_w'][i]
                h_s = self.parse_df['scaled_bbox_h'][i]
                mean_iou += self._compute_iou(w_c,h_c,w_s,h_s)
            mean_iou = mean_iou/sample 
            WCSS.append(wcss)
            MIOU.append(mean_iou)
            self.kmeans_map[k] = {
                "kmeans": kmeans,
                "mean_iou": mean_iou
            }

        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("Find Optimal Cluster")
        axs[0].plot(range(1, K+1), WCSS, '.-')
        axs[0].set_xlabel('Number of clusters')
        axs[0].set_ylabel('WCSS')
        axs[1].plot(range(1, K+1), MIOU, '.-')
        axs[1].set_xlabel('Centroids')
        axs[1].set_ylabel('Mean IOU')
        fig.tight_layout(pad=3.0)
        fig.savefig("find_best_k.png")
        plt.show()
        
    def show_k(self,k):
        kmeans = self.kmeans_map[k]["kmeans"]
        miou = self.kmeans_map[k]["mean_iou"]
        print("Centroids: %s" % kmeans.cluster_centers_)
        print("Mean IOU: %s" % miou)
        plt.scatter(self.parse_df['scaled_bbox_w'], self.parse_df['scaled_bbox_h'], c=kmeans.labels_.astype(float))
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red')
        plt.title("K=%s Clustered Bboxes" % k)
        plt.savefig("k%s_clustered_bboxes.png" % k)
        plt.show()