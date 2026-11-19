# Exact six-block Shapley allocation of posterior median marginal R-squared.
args<-commandArgs(TRUE)
if(length(args)!=3)stop('Usage: Rscript shapley.R context.csv output seed')
script<-sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1]);here<-dirname(normalizePath(script))
source(file.path(here,'bayesian_models.R'))
d<-read.csv(args[1]);out<-args[2];dir.create(out,recursive=TRUE,showWarnings=FALSE)
stopifnot(nrow(d)==1881,!anyDuplicated(d$amc))
model<-cmdstan_model(file.path(here,'hierarchical.stan'));groups<-blocks(d,TRUE)
# Fix the same screened predictor set before fitting any coalition.
screen<-screen_correlations(d,unlist(groups))
groups<-lapply(groups,function(x)intersect(x,screen$retained))
write.csv(data.frame(removed=screen$removed),file.path(out,'screened_out.csv'),row.names=FALSE)
values<-numeric(64);definitions<-list()
for(mask in 0:63){
  included<-which(as.logical(intToBits(mask)[1:6]));terms<-unlist(groups[included],use.names=FALSE)
  f<-fit_hierarchy(d,'yield',terms,model,as.integer(args[3])+mask)
  values[mask+1]<-median(f$r2)
  definitions[[mask+1]]<-data.frame(mask=mask,blocks=paste(names(groups)[included],collapse=';'),r2=values[mask+1])
}
phi<-numeric(6)
for(j in 0:5)for(mask in 0:63){
  bit<-bitwShiftL(1L,j)
  if(bitwAnd(mask,bit)!=0)next
  size<-sum(as.integer(intToBits(mask)[1:6]));weight<-factorial(size)*factorial(5-size)/factorial(6)
  phi[j+1]<-phi[j+1]+weight*(values[bitwOr(mask,bit)+1]-values[mask+1])
}
stopifnot(abs(sum(phi)-(values[64]-values[1]))<1e-10)
write.csv(do.call(rbind,definitions),file.path(out,'coalitions.csv'),row.names=FALSE)
write.csv(data.frame(domain=names(groups),contribution=phi,share_percent=100*phi/(values[64]-values[1])),file.path(out,'table_S3.csv'),row.names=FALSE)
