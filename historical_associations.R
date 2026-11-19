# Long-term means, adjusted contrasts and Table S1.
args <- commandArgs(TRUE)
if(length(args)!=2) stop('Usage: Rscript historical_associations.R amc_means.csv output_directory')
d <- read.csv(args[1]); out <- args[2]; dir.create(out,recursive=TRUE,showWarnings=FALSE)
climate <- c('temperature','vapour','precipitation','wind')
stopifnot(nrow(d)==3115, !anyDuplicated(d$amc), all(is.finite(as.matrix(d[c('yield',climate)]))))
fit <- lm(reformulate(climate,response='yield'),d)
write.csv(summary(fit)$coefficients,file.path(out,'coefficients.csv'))
curves <- contrasts <- incremental <- single <- list()
for(v in climate){
  reduced <- lm(reformulate(setdiff(climate,v),response='yield'),d)
  sse <- sum(residuals(fit)^2); sr <- sum(residuals(reduced)^2)
  incremental[[v]] <- data.frame(variable=v,delta_r2=summary(fit)$r.squared-summary(reduced)$r.squared,
    partial_r2=(sr-sse)/sr,p=anova(reduced,fit)$`Pr(>F)`[2])
  nd <- as.data.frame(lapply(d[climate],function(x)rep(mean(x),101)))
  nd[[v]] <- seq(min(d[[v]]),max(d[[v]]),length.out=101)
  curves[[v]] <- data.frame(variable=v,x=nd[[v]],predict(fit,nd,interval='confidence',level=.95))
  lo <- hi <- as.data.frame(lapply(d[climate],mean)); lo[[v]]<-quantile(d[[v]],.1);hi[[v]]<-quantile(d[[v]],.9)
  g <- model.matrix(delete.response(terms(fit)),hi)-model.matrix(delete.response(terms(fit)),lo)
  delta <- as.numeric(g%*%coef(fit));se<-sqrt(as.numeric(g%*%vcov(fit)%*%t(g)));crit<-qt(.975,df.residual(fit))
  contrasts[[v]] <- data.frame(variable=v,estimate=delta,lower=delta-crit*se,upper=delta+crit*se)
  for(mode in c('single','adjusted')){
    m<-if(mode=='single')lm(reformulate(v,response='yield'),d) else fit
    ratio<-sd(d[[v]])/sd(d$yield);ci<-confint(m,v)*ratio
    single[[paste(v,mode)]]<-data.frame(variable=v,model=mode,beta=coef(m)[v]*ratio,lower=ci[1],upper=ci[2])
  }
}
for(n in c('curves','contrasts','incremental','single'))write.csv(do.call(rbind,get(n)),file.path(out,paste0(n,'.csv')),row.names=FALSE)
saveRDS(fit,file.path(out,'long_term_model.rds'))
