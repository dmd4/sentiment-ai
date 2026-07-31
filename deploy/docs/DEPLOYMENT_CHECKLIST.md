# Deployment Checklist

## ✅ Pre-Deployment Steps

### **Code Preparation**
- [ ] All model files are in the repository
- [ ] Environment variables are properly configured
- [ ] Docker container builds successfully locally
- [ ] API responds to health checks
- [ ] All tests pass

### **Configuration**
- [ ] `MODEL_PATH` points to correct model file
- [ ] `TOKENIZER_PATH` points to correct tokenizer
- [ ] `MAX_SEQUENCE_LENGTH` is set to 22
- [ ] Port configuration matches deployment platform

### **Testing**
- [ ] Local Docker container works: `docker-compose up`
- [ ] API endpoints respond correctly
- [ ] Performance tests pass
- [ ] Integration tests pass

## 🚀 Deployment Options

### **Railway (Recommended for Demo)**
- ✅ Free tier available
- ✅ Automatic HTTPS
- ✅ Easy Docker deployment
- ✅ Built-in monitoring

### **Render**
- ✅ Free tier (with limitations)
- ✅ GitHub integration
- ✅ Automatic deployments

### **AWS ECS Fargate**
- ✅ Production-grade
- ✅ Auto-scaling
- ⚠️ Requires AWS knowledge
- ⚠️ Costs money

### **Google Cloud Run**
- ✅ Serverless
- ✅ Pay-per-use
- ✅ Auto-scaling
- ⚠️ Requires GCP setup

### **Kubernetes (Kustomize / Helm)**
- ✅ Production-grade cloud orchestration (EKS, GKE, AKS, Kind)
- ✅ Auto-scaling, self-healing, and load balancing
- ✅ Helm package management (`deploy/helm/sentiment-ai`)
- ⚠️ Requires Kubernetes cluster

### **Argo CD (GitOps)**
- ✅ Continuous delivery & automated cluster sync from Git
- ✅ Automatic drift detection and self-healing
- ✅ Declarative manifest (`deploy/argocd/application.yaml`)
- ⚠️ Requires Argo CD installed on cluster


## 📊 Post-Deployment Testing

```bash
# Test health endpoint
curl https://your-app.railway.app/health

# Test sentiment analysis
curl -X POST https://your-app.railway.app/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was amazing!"}'

# Test negation handling
curl -X POST https://your-app.railway.app/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was not bad"}'
```

## 🔍 Monitoring

- [ ] Health endpoint returns 200
- [ ] Response times < 100ms
- [ ] Error rates < 1%
- [ ] Memory usage stable
- [ ] No crashes or restarts
