# Production-context models and Tables S4-S8.
suppressPackageStartupMessages({library(cmdstanr);library(posterior)})
climate <- c('temperature','vapour','precipitation','wind')

sampling_ok <- function(fit){
  p<-fit$summary();q<-fit$diagnostic_summary()
  all(is.finite(p$rhat)) && max(p$rhat)<1.01 && min(p$ess_bulk)>=400 && min(p$ess_tail)>=400 &&
    all(q$ebfmi>.3) && sum(q$num_divergent)==0 && sum(q$num_max_treedepth)==0
}

fit_hierarchy <- function(d,outcome,predictors,model,seed){
  stopifnot(!anyDuplicated(d$amc),all(is.finite(as.matrix(d[c(outcome,predictors)]))))
  ym<-mean(d[[outcome]]);ys<-sd(d[[outcome]]);stopifnot(ys>0)
  x<-as.matrix(d[predictors]);xm<-colMeans(x);xs<-apply(x,2,sd)
  if(length(predictors) && any(xs==0))stop('Constant predictor in this sample')
  X<-if(length(predictors))cbind(intercept=1,scale(x,center=xm,scale=xs)) else matrix(1,nrow(d),1,dimnames=list(NULL,'intercept'))
  y<-(d[[outcome]]-ym)/ys
  U<-model.matrix(~factor(state)-1,d)
  V<-model.matrix(~interaction(state,intermediate,drop=TRUE)-1,d)
  H<-cbind(X,U,V)
  dat<-list(N=nrow(d),K=ncol(X),S=ncol(U),R=ncol(V),cross_x=crossprod(H),cross_y=as.vector(crossprod(H,y)),sum_y2=sum(y*y))
  fit<-model$sample(data=dat,seed=seed,chains=4,parallel_chains=4,iter_warmup=1000,iter_sampling=1000,adapt_delta=.99,max_treedepth=12,refresh=0)
  if(!sampling_ok(fit))fit<-model$sample(data=dat,seed=seed,chains=4,parallel_chains=4,iter_warmup=2000,iter_sampling=2000,adapt_delta=.999,max_treedepth=14,refresh=0)
  if(!sampling_ok(fit))stop('Sampling diagnostics failed after extended sampling')
  draws<-as_draws_matrix(fit$draws(c('beta','state_sd','region_sd','sigma')))
  beta<-draws[,paste0('beta[',seq_len(ncol(X)),']'),drop=FALSE];colnames(beta)<-colnames(X)
  variance<-rowSums((beta%*%cov(X))*beta)
  r2<-variance/(variance+draws[,'state_sd']^2+draws[,'region_sd']^2+draws[,'sigma']^2)
  list(beta=beta,r2=r2,x_mean=xm,x_sd=xs,y_mean=ym,y_sd=ys,diagnostics=fit$summary(),sampler=fit$diagnostic_summary())
}

summarize_fit <- function(f){
  q<-apply(f$beta,2,quantile,c(.025,.5,.975))
  data.frame(term=colnames(f$beta),estimate=q[2,],lower=q[1,],upper=q[3,],row.names=NULL)
}

blocks <- function(d,joint=FALSE){
  geography<-c(grep('^biome_',names(d),value=TRUE),'elevation')
  if(!joint)geography<-c(grep('^macroregion_',names(d),value=TRUE),geography)
  list(Climate=climate,Geography=geography,Structure=c('log_farm_size','log_cows'),
       Pasture=c('pasture_percentage','pasture_vigour'),Market=if(joint)'price_log_trend' else c('price_log_mean','price_log_trend'),
       Technology=c('mechanical_percentage','insemination_percentage','log1p_cooling_per_cow'))
}

screen_correlations <- function(d,terms,protected=climate){
  active<-terms;removed<-character()
  repeat{
    continuous<-active[vapply(d[active],function(x)length(unique(x))>2,logical(1))]
    if(length(continuous)<2)break
    rho<-cor(d[continuous],method='spearman')
    pairs<-which(upper.tri(rho)&abs(rho)>.70,arr.ind=TRUE)
    if(!nrow(pairs))break
    eligible<-apply(pairs,1,function(p)!all(colnames(rho)[p]%in%protected))
    pairs<-pairs[eligible,,drop=FALSE];if(!nrow(pairs))break
    pair<-colnames(rho)[pairs[which.max(abs(rho[pairs])),]]
    candidates<-setdiff(pair,protected)
    score<-sapply(candidates,function(v)mean(abs(rho[v,setdiff(colnames(rho),v)])))
    drop<-sort(candidates[score==max(score)])[1]
    active<-setdiff(active,drop);removed<-c(removed,drop)
  }
  list(retained=active,removed=removed)
}

run_domains <- function(d,model,seed,out){
  sets<-blocks(d);results<-list();index<-0
  for(y in c('yield','methane')){
    fits<-list()
    for(domain in c('baseline',setdiff(names(sets),'Climate'))){
      terms<-if(domain=='baseline')climate else c(climate,sets[[domain]])
      index<-index+1;f<-fit_hierarchy(d,y,terms,model,seed+index);fits[[domain]]<-f
      write.csv(summarize_fit(f),file.path(out,paste0(y,'_',domain,'.csv')),row.names=FALSE)
      saveRDS(f,file.path(out,paste0(y,'_',domain,'.rds')))
    }
    for(domain in names(fits))results[[paste(y,domain)]]<-data.frame(outcome=y,domain=domain,r2=median(fits[[domain]]$r2),delta_r2=median(fits[[domain]]$r2)-median(fits$baseline$r2))
  }
  write.csv(do.call(rbind,results),file.path(out,'domain_contributions.csv'),row.names=FALSE)
}

run_loso <- function(d,model,seed,out){
  sets<-blocks(d);rows<-list();index<-0
  for(state in unique(d$state))for(y in c('yield','methane')){
    q<-d[d$state!=state,];fits<-list()
    for(domain in c('baseline',setdiff(names(sets),'Climate'))){
      terms<-if(domain=='baseline')climate else c(climate,sets[[domain]])
      # An omitted state can remove every observation of a categorical level.
      terms<-terms[vapply(q[terms],sd,numeric(1))>0]
      index<-index+1;f<-fit_hierarchy(q,y,terms,model,seed+index);fits[[domain]]<-f
      adjusted<-f$beta
      for(v in terms)adjusted[,v]<-adjusted[,v]*f$y_sd/f$x_sd[v]*sd(d[[v]])/sd(d[[y]])
      f$beta<-adjusted;s<-summarize_fit(f);s<-s[s$term!='intercept',]
      rows[[length(rows)+1]]<-cbind(omitted_state=state,outcome=y,domain=domain,s)
    }
    for(domain in names(fits)){
      path<-file.path(out,paste(state,y,domain,'r2.csv',sep='_'))
      write.csv(data.frame(r2=median(fits[[domain]]$r2),delta_r2=median(fits[[domain]]$r2)-median(fits$baseline$r2)),path,row.names=FALSE)
    }
  }
  write.csv(do.call(rbind,rows),file.path(out,'loso_coefficients.csv'),row.names=FALSE)
}

run_drop_blocks <- function(d,model,seed,out){
  groups<-blocks(d,TRUE);screen<-screen_correlations(d,unlist(groups));terms<-screen$retained;rows<-list();index<-0
  write.csv(data.frame(removed=screen$removed),file.path(out,'screened_out.csv'),row.names=FALSE)
  for(y in c('yield','methane')){
    full<-fit_hierarchy(d,y,terms,model,seed)
    for(g in names(groups)){
      index<-index+1;f<-fit_hierarchy(d,y,setdiff(terms,groups[[g]]),model,seed+index)
      rows[[length(rows)+1]]<-data.frame(outcome=y,domain=g,delta_r2=median(full$r2)-median(f$r2))
    }
  }
  write.csv(do.call(rbind,rows),file.path(out,'drop_domain.csv'),row.names=FALSE)
}

supplementary_models <- function(){
  specifications<-list()
  add<-function(table,label,outcome,terms){specifications[[length(specifications)+1]]<<-list(table=table,label=label,outcome=outcome,terms=terms)}
  production<-list(farm='log_farm_size',cows='log_cows',both=c('log_farm_size','log_cows'))
  for(spec in c('four_climates','without_vapour')){
    cl<-if(spec=='four_climates')climate else setdiff(climate,'vapour')
    add('S4',paste(spec,'baseline'),'yield',cl)
    for(n in names(production))add('S4',paste(spec,n),'yield',c(cl,production[[n]]))
  }
  for(n in c('farm','cows'))add('S5',n,'methane',c(climate,production[[n]]))
  for(y in c('yield','methane')){
    for(v in c('pasture_percentage','pasture_vigour'))add('S6',paste(y,v),y,c(climate,v))
    add('S6',paste(y,'joint_with_cows'),y,c(climate,'pasture_percentage','pasture_vigour','log_cows'))
  }
  add('S7','climate','log_cows',c(climate,'pasture_percentage'))
  add('S7','climate_and_area','log_cows',c(climate,'pasture_percentage','log_area'))
  tech<-c('mechanical_percentage','insemination_percentage','log1p_cooling_total')
  for(v in tech)add('S8',v,'log_cows',c(climate,'log_area',v))
  add('S8','joint','log_cows',c(climate,'log_area',tech))
  specifications
}

run_supplementary <- function(d,model,seed,out){
  rows<-list();specs<-supplementary_models();stats<-list()
  for(i in seq_along(specs)){
    s<-specs[[i]];f<-fit_hierarchy(d,s$outcome,s$terms,model,seed+i)
    rows[[i]]<-cbind(table=s$table,model=s$label,outcome=s$outcome,summarize_fit(f))
    stats[[i]]<-data.frame(table=s$table,model=s$label,r2=median(f$r2))
  }
  write.csv(do.call(rbind,rows),file.path(out,'tables_S4_S8_coefficients.csv'),row.names=FALSE)
  statistics<-do.call(rbind,stats)
  statistics$delta_r2<-NA_real_
  for(spec in c('four_climates','without_vapour')){
    baseline<-statistics$r2[statistics$table=='S4' & statistics$model==paste(spec,'baseline')]
    take<-statistics$table=='S4' & startsWith(statistics$model,paste0(spec,' '))
    statistics$delta_r2[take]<-statistics$r2[take]-baseline
  }
  write.csv(statistics,file.path(out,'tables_S4_S8_r2.csv'),row.names=FALSE)
}

if(sys.nframe()==0){
  args<-commandArgs(TRUE)
  if(length(args)!=4)stop('Usage: Rscript bayesian_models.R context.csv output mode seed')
  d<-read.csv(args[1]);out<-args[2];dir.create(out,recursive=TRUE,showWarnings=FALSE)
  stopifnot(nrow(d)==1881,!anyDuplicated(d$amc))
  script<-sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1])
  model<-cmdstan_model(file.path(dirname(normalizePath(script)),'hierarchical.stan'))
  mode<-switch(args[3],domains=run_domains,loso=run_loso,drop=run_drop_blocks,supplementary=run_supplementary,stop('Unknown mode'))
  mode(d,model,as.integer(args[4]),out)
}
