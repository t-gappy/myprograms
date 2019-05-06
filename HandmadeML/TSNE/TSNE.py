import numpy as np

class TSNE(object):
    def __init__(self, lr=200., iter_num=1000, perplexity=32., out_dim = 2,
                    sigma=5.0, early_exaggeration = 4.):
        self.lr = lr
        self.iter_num = iter_num
        self.perplexity = perplexity
        self.sigma = sigma
        self.early_exaggeration = early_exaggeration
        self.out_dim = out_dim

    def transform(self, input_data):
        print("Calculating Joint Probability of Input Data..........", end="")
        proba_joint_x = self.calc_x_ij(input_data, self.sigma, self.perplexity)
        print("Done!")

        print("Preparing New Dimensional Data.......................")
        target = np.random.normal(0, 1e-4, (len(input_data), self.out_dim))
        target_hist = []
        target_hist.append(target)
        try:
            for i in range(self.iter_num):
                if i < 50:
                    pr_j_x = proba_joint_x * self.early_exaggeration
                    grad = self.calc_grad(pr_j_x, target)
                if i >= 50:
                    grad = self.calc_grad(proba_joint_x, target)
                proba_joint_y = self.calc_y_ij(target)
                loss = proba_joint_x * (np.log((proba_joint_x/(proba_joint_y+1e-30)+1e-30)))
                target = target - self.lr*grad + self.momentum(i)*(target - target_hist[np.maximum(0, i-1)])
                print("iter: {}, loss: {}".format(i, np.sum(loss)))
                target_hist.append(target)
            print("Accomplished!")

            return target

        except KeyboardInterrupt:
            return target


    def calc_perp(self, input_data, sigma, cond_index):
        """
        abstract:
            calculate conditional probability p(j|i) and perplexity with input sigma.
        input:
            input_data: what to transform
            sigma: use for calculation of conditional probability, std of Gaussian
            cond_index: conditioned index (j)
        output:
            perp_pred: calculated perplexity
            proba_cond: conditional probability of p(j|i)
        """
        # cond_indexにおける条件付き確率を求める
        norms = np.linalg.norm(input_data[cond_index] - input_data, axis = 1, ord=2) ** 2
        proba_cond = np.exp(-norms / (2*sigma**2))
        proba_cond[cond_index] = 0
        proba_cond /= np.sum(proba_cond)

        # Perplexity計算
        H = np.log2(proba_cond + 1e-8)
        H = H * proba_cond
        H = -np.sum(H)
        perp_pred = 2 ** H

        return perp_pred, proba_cond


    def binary_search(self, input_data, perplexity, sigma, cond_index):
        """
        abstract:
            search good sigma in condition 'i'.
        input:
            input_data: what to transform
            perplexity: range of cluster, use for searching sigma here
            sigma: sigma in search-start, maybe 5.0
            cond_index: conditioned index (j)
        output:
            proba_cond: conditional probability with perplexity-suitable sigma
        """
        cnt = 1
        perp_pred, _ = self.calc_perp(input_data, sigma, cond_index)
        while True:
            # 実数値の二分探索が不明だったので、大体収束しそうな100回で止める
            if perp_pred == perplexity or cnt >= 100:
                perp_pred, proba_cond = self.calc_perp(input_data, sigma, cond_index)
                return proba_cond
            elif perp_pred < perplexity:
                # Perplexityは5~50の範囲に存在する: Rangeが45
                sigma += (perplexity - perp_pred) / 45
                perp_pred, _ = self.calc_perp(input_data, sigma, cond_index)
            elif perp_pred > perplexity:
                sigma -= (perp_pred - perplexity) / 45
                perp_pred, _ = self.calc_perp(input_data, sigma, cond_index)
            cnt += 1


    def calc_x_ij(self, input_data, sigma, perplexity):
        """
        abstract:
            calculate joint probability p(i, j).
        input:
            input_data: what to transform
            sigma: sigma in search-start, maybe 5.0
            perplexity: range of cluster, use for searching good sigma
        output:
            proba_joint: joint probability p(i, j)
        """
        # 条件付き確率行列の作成
        proba_cond = []
        for i in range(len(input_data)):
            proba_cond_i = self.binary_search(input_data, perplexity, sigma, i)
            proba_cond.append(proba_cond_i)
        proba_cond = np.array(proba_cond)

        # 同時確率行列x(i, j)の作成
        proba_joint = (proba_cond + proba_cond.T) / (2 * proba_cond.shape[0])

        return proba_joint


    def calc_y_ij(self, target):
        """
        abstract:
            calculate joint probability q(i, j)
        input:
            target: y coordinate
        output:
            proba_joint: joint probability q(i, j)
        """
        proba_joint = []
        for i in range(target.shape[0]):
            norms = np.linalg.norm(target[i] - target, axis = 1, ord=2) ** 2
            proba_cond = (1+norms) ** -1
            proba_cond[i] = 0
            proba_joint.append(proba_cond)
        proba_joint = np.array(proba_joint)
        proba_joint /= np.sum(proba_joint)
        return proba_joint


    def calc_grad(self, proba_joint_x, target):
        proba_joint_y = self.calc_y_ij(target)
        differ_proba = proba_joint_x - proba_joint_y

        differ_vector = []
        norm_vector = []
        for i in range(len(target)):
            differ_vector_on_i = target[i] - target
            norms = np.linalg.norm(target[i] - target, axis = 1, ord=2) ** 2
            norm_vector_on_i = (1 + norms) ** -1
            differ_vector.append(differ_vector_on_i)
            norm_vector.append(norm_vector_on_i)
        differ_vector = np.array(differ_vector)
        norm_vector = np.array(norm_vector)
        grad = differ_proba[..., None] * differ_vector * norm_vector[..., None]
        grad = 4 * np.sum(grad, axis=1)

        return grad


    def momentum(self, iter_num):
        if iter_num >= 250:
            return 0.8
        else:
            return 0.5
