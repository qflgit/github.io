// ===== AI Knowledge Hub - 交互逻辑 =====

// 动态粒子背景
document.addEventListener('DOMContentLoaded', () => {
  const particlesContainer = document.querySelector('.particles');
  const particleCount = 25;

  for (let i = 0; i < particleCount; i++) {
    const particle = document.createElement('div');
    particle.className = 'particle';
    const size = Math.random() * 6 + 2;
    particle.style.width = size + 'px';
    particle.style.height = size + 'px';
    particle.style.left = Math.random() * 100 + '%';
    particle.style.top = Math.random() * 100 + '%';
    particle.style.animationDuration = (Math.random() * 10 + 10) + 's';
    particle.style.animationDelay = (Math.random() * 5) + 's';
    particle.style.background = ['#6366f1', '#f472b6', '#22d3ee'][Math.floor(Math.random() * 3)];
    particlesContainer.appendChild(particle);
  }
});

// 搜索功能
const searchInput = document.getElementById('searchInput');
const cards = document.querySelectorAll('.card[data-searchable]');

searchInput.addEventListener('input', (e) => {
  const query = e.target.value.toLowerCase();
  cards.forEach(card => {
    const title = card.querySelector('h3').textContent.toLowerCase();
    const desc = card.querySelector('p').textContent.toLowerCase();
    if (title.includes(query) || desc.includes(query)) {
      card.style.display = '';
      card.style.opacity = '1';
    } else {
      card.style.opacity = '0';
      setTimeout(() => { card.style.display = 'none'; }, 300);
    }
  });
});

// 分类筛选
const tabs = document.querySelectorAll('.tab');
const cardGrid = document.querySelector('.card-grid');

const topicsData = [
  { id: 'llm', title: '大语言模型', desc: 'GPT、Claude、Kimi 等 LLM 原理与应用', icon: '🧠', tag: 'hot', tagClass: 'tag-hot', category: 'foundation' },
  { id: 'agent', title: 'AI Agent', desc: '智能体架构、工具调用与自主决策系统', icon: '🤖', tag: 'new', tagClass: 'tag-new', category: 'agent' },
  { id: 'rag', title: 'RAG 检索增强', desc: '向量数据库、Embedding 与知识库构建', icon: '🔍', tag: '', tagClass: '', category: 'foundation' },
  { id: 'diffusion', title: '扩散模型', desc: 'Stable Diffusion、图像生成与视频合成', icon: '🎨', tag: 'hot', tagClass: 'tag-hot', category: 'multimodal' },
  { id: 'finetune', title: '模型微调', desc: 'LoRA、QLoRA、全参数微调与领域适配', icon: '⚙️', tag: '', tagClass: '', category: 'foundation' },
  { id: 'deploy', title: '模型部署', desc: 'vLLM、TensorRT、量化压缩与推理优化', icon: '🚀', tag: 'pro', tagClass: 'tag-pro', category: 'engineering' },
  { id: 'mcp', title: 'MCP 协议', desc: 'Model Context Protocol 工具集成标准', icon: '🔗', tag: 'new', tagClass: 'tag-new', category: 'agent' },
  { id: 'vision', title: '计算机视觉', desc: '目标检测、OCR、图像理解与多模态融合', icon: '👁️', tag: '', tagClass: '', category: 'multimodal' },
  { id: 'safety', title: 'AI 安全对齐', desc: 'RLHF、红队测试、偏见检测与内容审核', icon: '🛡️', tag: '', tagClass: '', category: 'foundation' },
  { id: 'chip', title: 'AI 芯片架构', desc: 'NPU、GPU 集群、存算一体与边缘推理', icon: '💾', tag: 'pro', tagClass: 'tag-pro', category: 'engineering' },
];

function renderCards(filter = 'all') {
  const filtered = filter === 'all' ? topicsData : topicsData.filter(t => t.category === filter);

  // 保留添加按钮之前的卡片
  const addCard = document.querySelector('.add-card');
  cardGrid.innerHTML = '';

  filtered.forEach((topic, index) => {
    const card = document.createElement('a');
    card.className = 'card';
    card.href = `subpages/${topic.id}.html`;
    card.setAttribute('data-searchable', '');
    card.style.animationDelay = (index * 0.05) + 's';

    const tagHtml = topic.tag ? `<span class="card-tag ${topic.tagClass}">${topic.tag}</span>` : '';

    card.innerHTML = `
      ${tagHtml}
      <div class="card-icon">${topic.icon}</div>
      <h3>${topic.title}</h3>
      <p>${topic.desc}</p>
      <div class="card-meta">
        <span>📄 12 篇文章</span>
        <span>⏱️ 更新 2 天前</span>
      </div>
    `;
    cardGrid.appendChild(card);
  });

  cardGrid.appendChild(addCard);
}

tabs.forEach(tab => {
  tab.addEventListener('click', () => {
    tabs.forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    renderCards(tab.dataset.filter);
  });
});

// 添加新主题
function addNewTopic() {
  const title = prompt('请输入新主题名称：');
  if (!title) return;
  const desc = prompt('请输入主题描述：') || '暂无描述';
  const icon = prompt('请输入图标 emoji（如 🎯）：') || '📌';

  const newTopic = {
    id: 'topic-' + Date.now(),
    title,
    desc,
    icon,
    tag: 'new',
    tagClass: 'tag-new',
    category: 'all'
  };

  topicsData.push(newTopic);
  renderCards('all');

  // 自动创建子页面模板
  createSubpageTemplate(newTopic);

  alert(`✅ 已添加主题「${title}」并生成子页面模板！`);
}

// 创建子页面模板
function createSubpageTemplate(topic) {
  const subpageHtml = `<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${topic.title} - AI Knowledge Hub</title>
  <link rel="stylesheet" href="../style.css">
  <style>
    .subpage-header { padding: 3rem 2rem 2rem; text-align: center; }
    .subpage-header h1 { font-size: 2.5rem; margin-bottom: 1rem; }
    .subpage-header h1 span { background: var(--gradient-1); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .article-list { max-width: 900px; margin: 0 auto; padding: 0 2rem 4rem; }
    .article-item {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 1.5rem;
      margin-bottom: 1rem;
      display: flex;
      align-items: center;
      gap: 1rem;
      transition: all 0.3s;
      cursor: pointer;
    }
    .article-item:hover {
      transform: translateX(8px);
      border-color: var(--primary-light);
      box-shadow: var(--shadow);
    }
    .article-num {
      width: 40px; height: 40px;
      border-radius: 12px;
      background: var(--gradient-1);
      display: flex; align-items: center; justify-content: center;
      font-weight: 700; font-size: 1.1rem;
    }
    .article-content h4 { font-size: 1.1rem; margin-bottom: 0.25rem; }
    .article-content p { color: var(--text-muted); font-size: 0.9rem; }
    .back-btn {
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.75rem 1.5rem;
      border-radius: 12px;
      background: var(--bg-card);
      border: 1px solid var(--border);
      color: var(--text);
      text-decoration: none;
      font-weight: 600;
      transition: all 0.3s;
      margin: 2rem;
    }
    .back-btn:hover {
      background: var(--gradient-1);
      border-color: transparent;
      transform: translateY(-2px);
    }
    .add-article {
      border: 2px dashed var(--border);
      background: transparent;
      justify-content: center;
      color: var(--text-muted);
    }
    .add-article:hover {
      border-color: var(--primary-light);
      color: var(--text);
    }
  </style>
</head>
<body>
  <div class="particles"></div>

  <header>
    <nav>
      <div class="logo">
        <div class="logo-icon">🧠</div>
        AI Knowledge Hub
      </div>
      <ul class="nav-links">
        <li><a href="../index.html">🏠 首页</a></li>
        <li><a href="../index.html#topics">📚 知识库</a></li>
        <li><a href="#" class="active">${topic.icon} ${topic.title}</a></li>
      </ul>
    </nav>
  </header>

  <a href="../index.html" class="back-btn">← 返回首页</a>

  <section class="subpage-header">
    <h1>${topic.icon} <span>${topic.title}</span></h1>
    <p>${topic.desc}</p>
  </section>

  <section class="article-list">
    <div class="article-item" onclick="alert('点击编辑此文章')">
      <div class="article-num">1</div>
      <div class="article-content">
        <h4>基础概念介绍</h4>
        <p>${topic.title} 的核心定义与发展历程</p>
      </div>
    </div>
    <div class="article-item" onclick="alert('点击编辑此文章')">
      <div class="article-num">2</div>
      <div class="article-content">
        <h4>关键技术原理</h4>
        <p>深入理解 ${topic.title} 的技术架构</p>
      </div>
    </div>
    <div class="article-item" onclick="alert('点击编辑此文章')">
      <div class="article-num">3</div>
      <div class="article-content">
        <h4>实践应用案例</h4>
        <p>真实场景中的 ${topic.title} 应用</p>
      </div>
    </div>
    <div class="article-item add-article" onclick="addArticle()">
      <div style="font-size: 1.5rem;">+</div>
      <div class="article-content">
        <h4>添加新文章</h4>
        <p>点击添加更多内容</p>
      </div>
    </div>
  </section>

  <footer>
    <p>🧠 AI Knowledge Hub — 探索人工智能的无限可能</p>
    <p style="margin-top: 0.5rem; font-size: 0.85rem;">使用此模板可自由扩展更多子话题</p>
  </footer>

  <script src="../script.js"></script>
  <script>
    function addArticle() {
      const title = prompt('请输入文章标题：');
      if (title) {
        alert('✅ 文章「' + title + '」已添加（实际项目中可连接后端保存）');
      }
    }
  </script>
</body>
</html>`;

  // 通过 fetch 请求后端创建文件（实际项目中）
  // 这里仅做演示，真实环境需要后端支持
  console.log(`子页面模板已生成: subpages/${topic.id}.html`);
}

// 粒子动画初始化
document.addEventListener('DOMContentLoaded', () => {
  renderCards('all');
});
