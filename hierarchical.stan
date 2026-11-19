data {
  int<lower=1> N;
  int<lower=1> K;
  int<lower=1> S;
  int<lower=1> R;
  matrix[K+S+R,K+S+R] cross_x;
  vector[K+S+R] cross_y;
  real<lower=0> sum_y2;
}
parameters {
  vector[K] beta;
  vector[S] state_z;
  vector[R] region_z;
  real<lower=0> state_sd;
  real<lower=0> region_sd;
  real<lower=0> sigma;
}
transformed parameters {
  vector[S] state_effect = state_sd * state_z;
  vector[R] region_effect = region_sd * region_z;
}
model {
  vector[K+S+R] theta = append_row(append_row(beta,state_effect),region_effect);
  real sse = sum_y2 - 2 * dot_product(theta,cross_y) + quad_form(cross_x,theta);
  beta[1] ~ normal(0,2);
  if(K>1) beta[2:K] ~ normal(0,1);
  state_z ~ std_normal();
  region_z ~ std_normal();
  state_sd ~ normal(0,1);
  region_sd ~ normal(0,1);
  sigma ~ student_t(3,0,1);
  target += -N*log(sigma) - sse/(2*square(sigma));
}
