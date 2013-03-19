// Alessandro Gentilini

#include "opencv2/core/core.hpp"

// Estimation of an affine 2D transformation by means of least squares method.
// Reference:
// SPÄTH, Helmuth. Fitting affine and orthogonal transformations between two sets of points. Mathematical Communications, 2004, 9.1: 27-34.
// http://hrcak.srce.hr/file/1425

template < typename point2D_t, typename point3D_t >
class LeastSquare2DAffineTransformationEstimator
{
public:
   // Solves the linear systems descripted by formula (17)
   static cv::Mat estimate( const std::vector<point2D_t>& P, const std::vector<point2D_t>& Q )
   {
      auto Q_tilde = Q_set_to_Q_matrix_tilde(P);

      auto c_tilde_0 = c_j_tilde(0,P,Q);
      auto c_tilde_1 = c_j_tilde(1,P,Q);

      auto Q_tilde_inv = Q_tilde.inv();
      auto a_tilde_0 = Q_tilde_inv * c_tilde_0;
      auto a_tilde_1 = Q_tilde_inv * c_tilde_1;

      cv::Mat t = cv::Mat::zeros( 3, 2, cv::DataType<point2D_t::value_type>::type );
      cv::Mat(a_tilde_0).copyTo(t.col(0));
      cv::Mat(a_tilde_1).copyTo(t.col(1));
      cv::transpose(t,t);
      return t;
   }

private:
   // Implements the formula (12)
   static cv::Mat q_to_q_tilde( const point2D_t& q )
   {
      return cv::Mat( point3D_t( q.x, q.y, 1 ) );
   }

   // Implements the formula (14)
   static cv::Mat Q_set_to_Q_matrix_tilde( const std::vector<point2D_t>& Q_set )
   {
      size_t m = Q_set.size();

      cv::Mat Q_matrix_tilde = cv::Mat::zeros( 3, 3, cv::DataType<point2D_t::value_type>::type );
      cv::Mat q_tilde;
      cv::Mat q_tilde_transposed;
      for ( size_t i = 0; i < m; i++ ) {
         q_tilde = q_to_q_tilde(Q_set[i]);
         cv::transpose( q_tilde, q_tilde_transposed );
         Q_matrix_tilde += q_tilde * q_tilde_transposed;
      }
      return Q_matrix_tilde;
   }

   // Implements the formula (16)
   static cv::Mat c_j_tilde( const size_t& j, const std::vector<point2D_t>& Q_set, const std::vector<point2D_t>& P_set )
   {
      if ( Q_set.size() != P_set.size() ) {
         throw 0;
      }

      if ( j > 2 ) {
         throw 1;
      }

      size_t m = Q_set.size();

      point2D_t::value_type p_ji;

      point2D_t::value_type c_j0 = 0;
      for ( size_t i = 0; i < m; i++ ) {
         switch( j ) {
         case 0: p_ji = P_set[i].x; break;
         case 1: p_ji = P_set[i].y; break;
         }
         c_j0 += Q_set[i].x * p_ji;
      }

      point2D_t::value_type c_j1 = 0;
      for ( size_t i = 0; i < m; i++ ) {
         switch( j ) {
         case 0: p_ji = P_set[i].x; break;
         case 1: p_ji = P_set[i].y; break;
         }
         c_j1 += Q_set[i].y * p_ji;
      }

      point2D_t::value_type c_j2 = 0;
      for ( size_t i = 0; i < m; i++ ) {
         switch( j ) {
         case 0: p_ji = P_set[i].x; break;
         case 1: p_ji = P_set[i].y; break;
         }
         c_j2 += 1 * p_ji;
      }

      return cv::Mat( point3D_t(c_j0,c_j1,c_j2) );
   }


};

#include <vector>
#include <iostream>

int main( int argc, char** argv )
{
   std::vector<cv::Point2f> P,Q;

   P.push_back(cv::Point2f( 1, 0));
   P.push_back(cv::Point2f( 0, 1));
   P.push_back(cv::Point2f(-1, 0));
   P.push_back(cv::Point2f( 0,-1));

   Q.push_back(cv::Point2f(1+sqrtf(2)/2, 1+sqrtf(2)/2));
   Q.push_back(cv::Point2f(1-sqrtf(2)/2, 1+sqrtf(2)/2));
   Q.push_back(cv::Point2f(1-sqrtf(2)/2, 1-sqrtf(2)/2));
   Q.push_back(cv::Point2f(1+sqrtf(2)/2, 1-sqrtf(2)/2));

   std::cout << LeastSquare2DAffineTransformationEstimator<cv::Point2f,cv::Point3f>::estimate(P,Q);

   return 0;
}