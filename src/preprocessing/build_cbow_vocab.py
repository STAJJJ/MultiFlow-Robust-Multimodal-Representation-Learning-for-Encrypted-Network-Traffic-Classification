import pandas as pd
import numpy as np
import pickle
from gensim.models import Word2Vec
from gensim.corpora import Dictionary
import os
import pdb


class AttributeEmbedding:
    def __init__(self, csv_path, output_dir, prefix, embedding_dim=300, window=5):
        """
        构建 Attribute Value Dictionary + Embedding Matrix
        :param csv_path: 输入 CSV 路径
        :param output_dir: 输出路径目录
        :param prefix: 前缀名（如 'len' 或 'iat'）
        :param embedding_dim: embedding 维度
        :param window: Word2Vec 上下文窗口
        """
        import pdb
        self.embedding_dim = embedding_dim
        self.prefix = prefix
        self.output_dir = output_dir

        # 1️⃣ 加载语料
        self.corpus = self.load_corpus(csv_path)
        print(f"[INFO] Loaded corpus from {csv_path}, total {len(self.corpus)} sequences.")



        # 3️⃣ 训练 Word2Vec 模型
        self.model = Word2Vec(
            sentences=self.corpus,
            vector_size=embedding_dim,
            window=window,
            min_count=1,
            sg=0,
            workers=8
        )
        
        model_path = os.path.join(output_dir, f"{prefix}_word2vec.model")
        self.model.save(model_path)
        print(f"[INFO] Saved Word2Vec model to {model_path}")
        

        # 4️⃣ 构建 Embedding Matrix
        self.embedding_matrix = self.model.wv.vectors
        
        pad_embedding=np.random.normal(scale=0.01, size=(1,self.embedding_matrix.shape[1]))
        unk_embedding=np.random.normal(scale=0.01, size=(1,self.embedding_matrix.shape[1],))
  
        self.embedding_matrix=np.concatenate((self.embedding_matrix,pad_embedding,unk_embedding),axis=0)
        
        # 2️⃣ 构建词典
        self.dictionary = self.model.wv.key_to_index
        self.dictionary['PAD']=len(self.dictionary.keys())
        self.dictionary['UNK']=len(self.dictionary.keys())
        #pdb.set_trace()
        dict_path = os.path.join(output_dir, f"{prefix}_dict.pkl")
        with open(dict_path, "wb") as f:
            pickle.dump(self.dictionary, f)
        print(f"[INFO] Saved dictionary to {dict_path} (vocab_size={len(self.dictionary)})")
        #self.embedding_matrix = self.build_embedding_matrix()

        embedding_path = os.path.join(output_dir, f"{prefix}_embedding.npy")
        np.save(embedding_path, self.embedding_matrix)
        print(f"[INFO] Saved embedding matrix to {embedding_path}, shape={self.embedding_matrix.shape}")

    @staticmethod
    def load_corpus(csv_path):
        corpus = []
        with open(csv_path, 'r') as f:
            for line in f:
                tokens = line.strip().split(',')
                if tokens and len(tokens) > 0:
                    corpus.append(tokens[:40])  # 每个流最多40个包
        return corpus

    def build_embedding_matrix(self):
        vocab_size = len(self.dictionary)
        matrix = np.zeros((vocab_size, self.embedding_dim))
        for token, idx in self.dictionary.token2id.items():
            if token in self.model.wv:
                matrix[idx] = self.model.wv[token]
            else:
                # OOV token 生成随机向量
                matrix[idx] = np.random.normal(scale=0.01, size=(self.embedding_dim,))
        return matrix

    def get_vector(self, token):
        """
        查询某个 token 的 embedding
        :param token: 字符串，例如 "70" 或 "0.000002"
        :return: np.array 维度 (embedding_dim,)
        """
        if token in self.dictionary.keys():
            idx = self.dictionary[token]
            return self.embedding_matrix[idx]
        #elif token in self.model.wv:
        #    return self.model.wv[token]
        else:
            print(f"[WARNING] Token '{token}' OOV, returning random vector.")
            return np.random.normal(scale=0.01, size=(self.embedding_dim,))

# ================= 配置路径 =================
base_dir = "/3241903007/workstation/AnomalyTrafficDetection/FlowVocab/dataset/AttributeValueDictionary"

#os.makedirs(base_dir, exist_ok=True)

# ================= 构建 Length Attribute =================
length_embedder = AttributeEmbedding(
    csv_path=os.path.join(base_dir, "len_corpus_2_7.csv"),
    output_dir=base_dir,
    prefix="len",
    embedding_dim=300,
    window=4
)

# ================= 构建 IAT Attribute =================
iat_embedder = AttributeEmbedding(
    csv_path=os.path.join(base_dir, "iat_corpus_2_7.csv"),
    output_dir=base_dir,
    prefix="iat",
    embedding_dim=300,
    window=4
)

# ================= 查询示例 =================
vec_len_722 = length_embedder.get_vector("722")
vec_iat_000002 = iat_embedder.get_vector("0.000002")

print("Length(722) embedding 前5维:", vec_len_722[:5])
print("IAT(0.000002) embedding 前5维:", vec_iat_000002[:5])
print("Length(722) embedding 维度:", vec_len_722.shape)
print("IAT(0.000002) embedding 维度:", vec_iat_000002.shape)
