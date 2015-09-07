#ifndef _0614ICP_ICP_HPP_
#define _0604ICP_ICP_HPP_

#include <iostream>
#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/LU>
#include <opencv2/opencv.hpp>

template <typename CorrespondenceEstimation, typename TransformSolver, typename Scalar>
class ICP
{
public:
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixX;
	typedef Eigen::Matrix<Scalar, 3, 3> RotMatrix;
	typedef Eigen::Matrix<Scalar, 3, 1> TransVec;

	//コンストラクタ
	ICP(int max_iter = 100, Scalar dif_epsilon = 1e-6, Scalar transform_epsilon = 1e-6)
		: max_iter_(max_iter), dif_epsilon_(dif_epsilon), transform_epsilon_(transform_epsilon), vis_rows_(480), vis_cols_(640), vis_focal_(100)
	{}

	void setInputSource( const MatrixX& source ) { source_ = source.transpose(); }

	void setInputTarget( const MatrixX& target ) { target_ = target.transpose(); }

	//位置合わせ
	void align(bool visualize = true)
	{
		int n_source = static_cast<int>(source_.cols());
		int n_target = static_cast<int>(target_.cols());

		//初期値を入れる
		R_ = RotMatrix::Identity(); //単位行列
		t_ = TransVec::Zero(); //ゼロベクトル

		//P0
		MatrixX source_transformed = R_ * source_ + t_.replicate(1, n_source);

		//繰り返し数の初期化
		iter_ = 0;
		dif_ = std::numeric_limits<Scalar>::max();

		while (iter_ < max_iter_)
		{
			// Step1: 最近傍点の計算
			MatrixX correspondences;
			corr_estimation_.search(source_transformed, target_, correspondences);

			// Step2: 位置合わせ
			RotMatrix Ri;
			TransVec ti;
			trans_solver_.solve(source_transformed, correspondences, Ri, ti);

			//途中経過の観察
			if (visualize)
				visualizeIn2D(source_transformed, correspondences);

			// Step3: 変換の適用
			source_transformed = Ri * source_transformed + ti.replicate(1, n_source);

			std::cout << "R: " << R_ << std::endl;
			std::cout << "t: " << t_ << std::endl;

			//途中経過の観察
			if (visualize)
				visualizeIn2D(source_transformed, correspondences);

			// Step4: 収束判定
			Scalar tmp = 0;
			//corr_estimation_.search(source_transformed, target_, correspondences);
			for (int i = 0; i < n_source; ++i)
			{
				tmp += ( source_transformed.col(i) - target_.col(i) ).norm();
			}
			tmp = (1.0 / n_source) * tmp;

			if (abs(dif_ - tmp) <  dif_epsilon_)
				break;
			else if ((Ri.norm() + ti.norm()) < transform_epsilon_)
				break;

			dif_ = tmp;

			R_ = Ri * R_;
			t_ = R_ * ti + t_;

			iter_++;
		}
	}


private:

	void visualizeIn2D(const MatrixX& source_transformed, const MatrixX& correspondences) const
	{
		cv::Mat vis(vis_rows_, vis_cols_, CV_8UC3, cv::Scalar(0, 0, 0));

		Scalar cx = vis_cols_ / 2.0 - 0.5;
		Scalar cy = vis_rows_ / 2.0 - 0.5;

		int shape_size = 3;

		auto round = [](Scalar number) -> int {
			return (static_cast<int>((number < 0.0) ? std::ceil(number - 0.5) : std::floor(number + 0.5)));
		};

		//変換結果
		for (int i = 0; i < source_transformed.cols(); ++i)
		{
			TransVec pt = source_transformed.col(i);

			//画像平面へ投影
			Scalar u = vis_focal_ * (pt.x() / pt.z()) + cx;
			Scalar v = vis_focal_ * (pt.y() / pt.z()) + cy;

			//整数に丸めて、その位置に赤丸を描画
			cv::circle(vis, cv::Point(round(u), round(v)), shape_size, cv::Scalar(0, 0, 255), CV_FILLED);
		}

		//source
		for (int i = 0; i < source_.cols(); ++i)
		{
			TransVec pt = source_.col(i);

			//画像平面へ投影
			Scalar u = vis_focal_ * (pt.x() / pt.z()) + cx;
			Scalar v = vis_focal_ * (pt.y() / pt.z()) + cy;

			//整数に丸めて、その位置に赤丸を描画
			cv::circle(vis, cv::Point(round(u), round(v)), shape_size * 2, cv::Scalar(255, 0, 0), CV_FILLED);
		}

		//target
		for (int i = 0; i < target_.cols(); ++i)
		{
			TransVec pt = target_.col(i);

			//画像平面へ投影
			Scalar u = vis_focal_ * (pt.x() / pt.z()) + cx;
			Scalar v = vis_focal_ * (pt.y() / pt.z()) + cy;

			//整数に丸めて、その位置に赤丸を描画
			cv::circle(vis, cv::Point(round(u), round(v)), shape_size * 2, cv::Scalar(0, 255, 0), CV_FILLED);
		}

		//対応点
		for (int i = 0; i < correspondences.cols(); ++i)
		{
			TransVec pt = correspondences.col(i);
			TransVec pt2 = source_transformed.col(i);

			//画像平面へ投影
			Scalar u = vis_focal_ * (pt.x() / pt.z()) + cx;
			Scalar v = vis_focal_ * (pt.y() / pt.z()) + cy;
			Scalar u2 = vis_focal_ * (pt2.x() / pt2.z()) + cx;
			Scalar v2 = vis_focal_ * (pt2.y() / pt2.z()) + cy;

			//整数に丸めて、その位置に赤丸を描画
			cv::line(vis, cv::Point(round(u), round(v)), cv::Point(round(u2), round(v2)), cv::Scalar(0, 255, 255));
		}

		cv::imshow("alignment", vis);
		cv::waitKey(0);

	}

	// 対応点探索
	CorrespondenceEstimation corr_estimation_;

	// 変換推定
	TransformSolver trans_solver_;

	// ソース (データ点)
	MatrixX source_;

	// ターゲット (モデル点)
	MatrixX target_;

	//回転行列
	RotMatrix R_;

	//並進ベクトル
	TransVec t_;

	//繰り返し数
	int iter_;

	//最大繰り返し数
	int max_iter_;

	//二乗誤差
	Scalar dif_;

	//二乗誤差の減少閾値
	Scalar dif_epsilon_;

	//変換の減少閾値
	Scalar transform_epsilon_;

	//
	int vis_rows_;
	int vis_cols_;
	Scalar vis_focal_;
};

template <typename Scalar>
class ClosestPointLiner
{
public:

	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixX;

	void search(const MatrixX& source, const MatrixX& target, MatrixX& correspondences) const
	{
		correspondences.resize(source.rows(), source.cols());

		for (int i = 0; i < source.cols(); ++i)
		{
			int min_index = -1;
			Scalar min_dist = std::numeric_limits<Scalar>::max();

			for (int j = 0; j < target.cols(); ++j)
			{
				Scalar d = (source.col(i) - target.col(j)).norm();

				if (d < min_dist)
				{
					min_dist = d;
					min_index = j;
				}
			}

			correspondences.col(i) = target.col(min_index);
		}
	}

	void operator () (const MatrixX& source, const MatrixX& target, MatrixX& correspondences)
	{
		// サイズをソースに合わせる
		correspondences.resize(source.rows(), source.cols());

		for (int i = 0; i < source.rows(); ++i)
		{
			// 二重ループとやることは同じ
			MatrixX difference = target - source.row(i).replicate(target.rows(), 1);

			// 行列要素ごとの二乗
			difference = difference.array().pow(2);

			// 行方向への和 (sqrt とってない距離)
			MatrixX distance = difference.rowwise().sum();

			// 最小距離は何行目？
			MatrixX::Index min_row;
			distance.minCoeff(min_row, NULL);

			// 対応点として保持
			correspondences.row(i) = model.row(min_row);
		}
	}

};

template <typename Scalar>
class SolveTransformSVD
{
public:
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixX;
	typedef Eigen::Matrix<Scalar, 3, 3> RotMatrix;
	typedef Eigen::Matrix<Scalar, 3, 1> TransVec;

	//コンストラクタ
	SolveTransformSVD(void)
	{}

	//変換を求める
	void solve(const MatrixX& source, const MatrixX& correspondences, RotMatrix& R, TransVec& t) const
	{
		//データ数
		int n_data = static_cast<int>(source.cols()); // = correspondences.cols()

		//重心の位置
		MatrixX centroid_src = source.rowwise().mean();
		MatrixX centroid_corr = correspondences.rowwise().mean();

		//重心座標
		MatrixX centroid_coord_src = source - centroid_src.replicate(1, n_data);
		MatrixX centroid_coord_corr = correspondences - centroid_corr.replicate(1, n_data);

		RotMatrix covariance = RotMatrix::Zero();

		for (int i = 0; i < n_data; ++i)
		{
			covariance += centroid_coord_src.col(i) * centroid_coord_corr.col(i).transpose();
		}

		//共分散行列
		covariance = (1.0 /n_data) * covariance;

		//特異値分解
		Eigen::JacobiSVD<RotMatrix> svd(covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);

		RotMatrix UV = svd.matrixU() * svd.matrixV().transpose();

		RotMatrix sign;
		sign <<
			1, 0, 0,
			0, 1, 0,
			0, 0, UV.determinant();

		//回転行列
		R = (svd.matrixU() * sign * svd.matrixV().transpose()).transpose();

		//並進ベクトル
		t = centroid_corr - R * centroid_src;
	}

};

#endif // _0614ICP_ICP_HPP_
