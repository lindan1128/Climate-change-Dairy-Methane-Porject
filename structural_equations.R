# SEM estimation, path effects, temporal windows and random AMC subsets.
suppressPackageStartupMessages(library(lavaan))
climate_nodes<-c('temperature','vapour','precipitation','wind')
directed_equations<-paste(c(
  paste('P ~ G +',paste(climate_nodes,collapse=' + ')),
  'S ~ M',
  paste('Y ~ T + S +',paste(climate_nodes,collapse=' + '),'+ P + N + G'),
  paste('E ~ P +',paste(climate_nodes,collapse=' + '),'+ N + Y + G + M'),
  paste(climate_nodes,'~ G')),collapse='\n')

fit_sem<-function(d,syntax){
  fit<-sem(syntax,data=d,estimator='ML',meanstructure=TRUE,fixed.x=FALSE,auto.cov.y=FALSE)
  if(!lavInspect(fit,'converged') || !lavInspect(fit,'post.check'))stop('SEM did not produce an admissible converged fit')
  fit
}

path_effects<-function(fit){
  p<-standardizedSolution(fit);p<-p[p$op=='~',]
  nodes<-unique(c(p$lhs,p$rhs));A<-matrix(0,length(nodes),length(nodes),dimnames=list(nodes,nodes))
  A[cbind(match(p$lhs,nodes),match(p$rhs,nodes))]<-p$est.std
  total<-A;power<-A
  for(k in 2:length(nodes)){power<-power%*%A;total<-total+power}
  if(max(abs(power))>1e-10)stop('Directed graph contains a cycle')
  rows<-list()
  for(target in intersect(c('Y','E'),nodes))for(source in setdiff(nodes,target)){
    rows[[length(rows)+1]]<-data.frame(source=source,target=target,direct=A[target,source],indirect=total[target,source]-A[target,source])
  }
  do.call(rbind,rows)
}

bollen_stine<-function(fit,syntax,seed){
  set.seed(seed);X<-lavInspect(fit,'data');S<-lavInspect(fit,'sampstat')$cov;implied<-fitted(fit)
  root<-function(A){e<-eigen(A,symmetric=TRUE);if(min(e$values)<=0)stop('Non-positive covariance');e$vectors%*%diag(sqrt(e$values))%*%t(e$vectors)}
  Z<-scale(X,center=TRUE,scale=FALSE)%*%root(solve(S))%*%root(implied$cov)
  Z<-sweep(Z,2,implied$mean,'+');colnames(Z)<-colnames(X)
  statistics<-rep(NA_real_,499)
  for(b in seq_len(499)){
    f<-tryCatch(fit_sem(as.data.frame(Z[sample.int(nrow(Z),replace=TRUE),]),syntax),error=function(e)NULL)
    if(!is.null(f))statistics[b]<-fitMeasures(f,'chisq')
  }
  good<-is.finite(statistics)
  if(!any(good))stop('No successful Bollen-Stine refits')
  list(p=mean(statistics[good]>=fitMeasures(fit,'chisq')),successful=sum(good),requested=499,statistics=statistics)
}

case_bootstrap<-function(d,syntax,seed){
  set.seed(seed);reference<-path_effects(fit_sem(d,syntax));draws<-list()
  for(b in seq_len(499)){
    f<-tryCatch(fit_sem(d[sample.int(nrow(d),replace=TRUE),],syntax),error=function(e)NULL)
    if(!is.null(f))draws[[length(draws)+1]]<-cbind(replicate=b,path_effects(f))
  }
  if(!length(draws))stop('No successful case-bootstrap fits')
  boot<-do.call(rbind,draws);rows<-list()
  for(i in seq_len(nrow(reference)))for(kind in c('direct','indirect')){
    q<-boot[boot$source==reference$source[i]&boot$target==reference$target[i],kind]
    ci<-quantile(q,c(.025,.975))
    rows[[length(rows)+1]]<-data.frame(source=reference$source[i],target=reference$target[i],kind=kind,
      estimate=reference[[kind]][i],lower=ci[1],upper=ci[2],successful=length(q),requested=499)
  }
  list(summary=do.call(rbind,rows),draws=boot)
}

compare_sem<-function(fit,reference){
  a<-standardizedSolution(fit);b<-standardizedSolution(reference)
  a<-a[a$op=='~',];b<-b[b$op=='~',]
  j<-match(paste(b$lhs,b$rhs),paste(a$lhs,a$rhs));stopifnot(!anyNA(j))
  delta<-a$est.std[j]-b$est.std
  c(direction_agreement=mean(sign(a$est.std[j])==sign(b$est.std)),beta_mae=mean(abs(delta)),
    fitMeasures(fit,c('chisq','rmsea','srmr','gfi')),yield_r2=lavInspect(fit,'r2')['Y'],methane_r2=lavInspect(fit,'r2')['E'])
}

sample_sensitivity<-function(d,syntax,seed){
  set.seed(seed);reference<-fit_sem(d,syntax);rows<-list()
  samples<-lapply(c(500,1000,1200,1500),function(n)replicate(100,sample.int(nrow(d),n,replace=FALSE),simplify=FALSE))
  for(i in seq_along(samples))for(j in 1:100){
    q<-d[samples[[i]][[j]],];f<-fit_sem(q,syntax);bs<-bollen_stine(f,syntax,seed+100*i+j)
    rows[[length(rows)+1]]<-data.frame(n=nrow(q),replicate=j,t(compare_sem(f,reference)),bootstrap_p=bs$p,bootstrap_successful=bs$successful)
  }
  do.call(rbind,rows)
}

window_inputs<-function(annual,static,start,affine){
  d<-annual[annual$year>=start&annual$year<=2024,]
  fields<-c('yield','methane','cows','pasture_percentage','log_price',climate_nodes)
  stopifnot(!anyDuplicated(d[c('amc','year')]),all(table(d$amc)==2025-start),all(is.finite(as.matrix(d[fields]))))
  means<-aggregate(d[fields],list(amc=d$amc),mean);means<-merge(means,static,by='amc',sort=FALSE)
  weight<-mean((start:2024)<=2017)
  raw<-data.frame(amc=means$amc,Y=log(means$yield),E=log(means$methane),N=log(means$cows),
    P=means$pasture_percentage,M=means$log_price,S=log(weight*means$farm_size_2006+(1-weight)*means$farm_size_2017),G=means$G,T=means$T)
  raw[climate_nodes]<-means[climate_nodes]
  for(v in affine$variable)raw[[v]]<-(raw[[v]]-affine$center[affine$variable==v])/affine$scale[affine$variable==v]
  raw
}

run_sem<-function(d,syntax,out,seed){
  dir.create(out,recursive=TRUE,showWarnings=FALSE);fit<-fit_sem(d,syntax)
  p<-standardizedSolution(fit)
  write.csv(p,file.path(out,'all_paths.csv'),row.names=FALSE)
  write.csv(p[p$op=='~'&p$pvalue<.05,],file.path(out,'display_paths.csv'),row.names=FALSE)
  saveRDS(fit,file.path(out,'model.rds'))
  write.csv(as.data.frame(t(fitMeasures(fit,c('chisq','rmsea','srmr','gfi')))),file.path(out,'fit.csv'),row.names=FALSE)
  r2<-lavInspect(fit,'r2')
  write.csv(data.frame(outcome=names(r2),r2=as.numeric(r2)),file.path(out,'outcome_r2.csv'),row.names=FALSE)
  saveRDS(bollen_stine(fit,syntax,seed),file.path(out,'bollen_stine.rds'))
  effect<-case_bootstrap(d,syntax,seed)
  write.csv(effect$summary,file.path(out,'effects.csv'),row.names=FALSE)
  write.csv(effect$draws,file.path(out,'effect_bootstrap.csv'),row.names=FALSE)
}

run_temporal_sensitivity<-function(annual,static,affine,syntax,seed){
  reference_data<-window_inputs(annual,static,2007,affine)
  reference<-fit_sem(reference_data,syntax);rows<-list()
  for(start in c(2007,2010,2015,2020)){
    d<-window_inputs(annual,static,start,affine);f<-fit_sem(d,syntax)
    bs<-bollen_stine(f,syntax,seed+start)
    rows[[as.character(start)]]<-data.frame(start=start,end=2024,t(compare_sem(f,reference)),bootstrap_p=bs$p,bootstrap_successful=bs$successful)
  }
  do.call(rbind,rows)
}

summarize_sample_sensitivity<-function(results){
  fields<-setdiff(names(results),c('n','replicate','bootstrap_p','bootstrap_successful'))
  rows<-list()
  for(n in unique(results$n)){
    d<-results[results$n==n,]
    for(v in fields)rows[[length(rows)+1]]<-data.frame(n=n,statistic=v,median=median(d[[v]]),minimum=min(d[[v]]),maximum=max(d[[v]]))
    rows[[length(rows)+1]]<-data.frame(n=n,statistic='bootstrap_p_at_least_0.05',median=sum(d$bootstrap_p>=.05),minimum=NA,maximum=NA)
  }
  do.call(rbind,rows)
}
