import { LeaderboardStore } from '../stores/LeaderboardStore.js';
import { SoundManager } from '../sound/SoundManager.js';

// 统一样式配置：确保文字美观性、一致性，避免遮挡
const STYLE_CONFIG = {
  // 基础字体：适配中英文，提升跨设备显示效果
  baseFont: "'PingFang SC', 'Microsoft YaHei', Arial, sans-serif",
  // 文字阴影：增强立体感，避免与深色背景融合
  textShadow: { offsetX: 1, offsetY: 1, color: '#00000080', blur: 2, fill: true },
  titleShadow: { offsetX: 2, offsetY: 2, color: '#00000090', blur: 4, fill: true },
  // 颜色体系：区分功能模块，提升视觉辨识度
  color: {
    title: '#ffffff',          // 页面标题色
    backBtn: '#00ffff',        // 返回按钮文字色
    backBtnHover: '#80ffff',   // 返回按钮悬浮色
    exportBtn: '#7cc7ff',      // 导出按钮文字色
    exportBtnHover: '#a8d8ff', // 导出按钮悬浮色
    importBtn: '#ffd85e',      // 导入按钮文字色
    importBtnHover: '#fff0a8', // 导入按钮悬浮色
    section: {                 // 排行榜板块文字色
      easy: '#79ff97',
      medium: '#7cc7ff',
      hard: '#ff9b7c',
      custom: '#d79bff'
    },
    rankText: '#ffffff',       // 排名文字色
    infoText: '#9bd',          // 图片信息文字色
    timeText: '#ffd700',       // 时间文字色
    emptyText: '#aaaaaa'       // 空记录文字色
  },
  // 按钮样式：统一尺寸、内边距、圆角
  button: {
    height: 36,
    paddingX: 16,
    paddingY: 8,
    borderRadius: 8,
    bg: {
      back: '#222222',
      backHover: '#333333',
      export: '#223040',
      exportHover: '#2d4058',
      import: '#403822',
      importHover: '#584e32'
    }
  },
  // 排行榜板块样式
  section: {
    width: '45%',    // 板块宽度（相对屏幕）
    minWidth: 280,   // 最小宽度（避免小屏挤压）
    height: 180,     // 固定高度
    padding: 16,     // 内边距
    bg: 0x131a20,    // 背景色
    border: 0x2b3642 // 边框色
  }
};

export class Leaderboard extends Phaser.Scene {
  constructor() {
    super('Leaderboard');
    this.uiDepth = 100; // 统一UI层级，确保文字不被遮挡
    this._toastNodes = null; // 存储Toast元素，用于销毁
  }

  preload() {
    // 预加载音频资源（避免播放时卡顿）
    const audios = [
      { key: 'bgm_menu', urls: ['assets/audio/bgm_menu.mp3'] },
      { key: 'click', urls: ['assets/audio/click.wav'] }
    ];
    audios.forEach(a => {
      if (!this.sound.get(a.key)) this.load.audio(a.key, a.urls);
    });
  }

  create() {
    SoundManager.init(this);
    SoundManager.playBGM('bgm_menu');

    const W = this.cameras.main.width;
    const H = this.cameras.main.height;
    const centerX = W / 2; // 屏幕水平中心（统一复用）
    const topMargin = 24;  // 顶部元素间距

    // 1. 背景层（兜底，避免透明区域导致文字显影异常）
    const bg = this.add.graphics().setDepth(this.uiDepth - 10);
    bg.fillStyle(0x0e1218, 1);
    bg.fillRect(0, 0, W, H);

    // 2. 页面标题（居中显示，增强视觉焦点）
    this.add.text(centerX, 60, '排行榜', {
      fontSize: '38px',
      color: STYLE_CONFIG.color.title,
      fontStyle: 'bold',
      fontFamily: STYLE_CONFIG.baseFont,
      shadow: STYLE_CONFIG.titleShadow
    }).setOrigin(0.5).setDepth(this.uiDepth);

    // 3. 顶部功能按钮组（返回、导出、导入，文字居中，防遮挡）
    this.createTopButtons(W, H, topMargin);

    // 4. 排行榜数据加载与渲染（板块居中，文字对齐）
    const allRecords = LeaderboardStore.getAll();
    this.sectionsConfig = [
      { title: '简单 3x3', key: 'EASY', color: STYLE_CONFIG.color.section.easy },
      { title: '中等 4x4', key: 'MEDIUM', color: STYLE_CONFIG.color.section.medium },
      { title: '困难 5x5', key: 'HARD', color: STYLE_CONFIG.color.section.hard },
      { title: '自定义', key: 'CUSTOM', color: STYLE_CONFIG.color.section.custom }
    ];
    this.drawSections(allRecords, W, H);
  }

  /**
   * 创建顶部功能按钮（返回、导出、导入）
   * @param {number} W - 屏幕宽度
   * @param {number} H - 屏幕高度
   * @param {number} topMargin - 顶部间距
   */
  createTopButtons(W, H, topMargin) {
    // 3.1 返回按钮（左上，文字居中，带交互反馈）
    const backBtn = this.createStyledButton({
      x: 50,
      y: topMargin + STYLE_CONFIG.button.height / 2,
      text: '返回',
      textColor: STYLE_CONFIG.color.backBtn,
      textColorHover: STYLE_CONFIG.color.backBtnHover,
      bgColor: STYLE_CONFIG.button.bg.back,
      bgColorHover: STYLE_CONFIG.button.bg.backHover,
      onClick: () => {
        SoundManager.playClick();
        this.scene.start('MainMenu');
      }
    });

    // 3.2 导出按钮（右上，文字居中）
    const exportBtn = this.createStyledButton({
      x: W - 180,
      y: topMargin + STYLE_CONFIG.button.height / 2,
      text: '导出记录',
      textColor: STYLE_CONFIG.color.exportBtn,
      textColorHover: STYLE_CONFIG.color.exportBtnHover,
      bgColor: STYLE_CONFIG.button.bg.export,
      bgColorHover: STYLE_CONFIG.button.bg.exportHover,
      onClick: () => {
        SoundManager.playClick();
        this.exportLeaderboard();
      }
    });

    // 3.3 导入按钮（右上，导出按钮右侧，文字居中）
    const importBtn = this.createStyledButton({
      x: W - 80,
      y: topMargin + STYLE_CONFIG.button.height / 2,
      text: '导入记录',
      textColor: STYLE_CONFIG.color.importBtn,
      textColorHover: STYLE_CONFIG.color.importBtnHover,
      bgColor: STYLE_CONFIG.button.bg.import,
      bgColorHover: STYLE_CONFIG.button.bg.importHover,
      onClick: () => {
        SoundManager.playClick();
        this.importLeaderboard();
      }
    });
  }

  /**
   * 创建带交互效果的Styled按钮（文字居中，防遮挡）
   * @param {object} options - 按钮配置
   * @returns {Phaser.GameObjects.Container} 按钮容器（背景+文字）
   */
  createStyledButton(options) {
    const { x, y, text, textColor, textColorHover, bgColor, bgColorHover, onClick } = options;

    // 1. 按钮文字（先计算文字宽度，用于适配背景）
    const textObj = this.add.text(0, 0, text, {
      fontSize: '20px',
      color: textColor,
      fontFamily: STYLE_CONFIG.baseFont,
      shadow: STYLE_CONFIG.textShadow
    }).setOrigin(0.5).setDepth(this.uiDepth + 1);

    // 2. 按钮背景（根据文字宽度适配，确保文字居中）
    const bgWidth = textObj.width + STYLE_CONFIG.button.paddingX * 2;
    const bgHeight = STYLE_CONFIG.button.height;
    const bgObj = this.add.graphics().setDepth(this.uiDepth);
    bgObj.fillStyle(bgColor, 1);
    bgObj.fillRoundedRect(-bgWidth / 2, -bgHeight / 2, bgWidth, bgHeight, STYLE_CONFIG.button.borderRadius);

    // 3. 按钮容器（整合背景和文字，确保整体居中）
    const btnContainer = this.add.container(x, y, [bgObj, textObj]).setDepth(this.uiDepth);
    // 交互区域（与背景大小一致）
    btnContainer.setInteractive(
      new Phaser.Geom.Rectangle(-bgWidth / 2, -bgHeight / 2, bgWidth, bgHeight),
      Phaser.Geom.Rectangle.Contains
    );

    // 4. 交互反馈（hover时变色，增强操作感）
    btnContainer.on('pointerover', () => {
      textObj.setColor(textColorHover);
      bgObj.clear();
      bgObj.fillStyle(bgColorHover, 1);
      bgObj.fillRoundedRect(-bgWidth / 2, -bgHeight / 2, bgWidth, bgHeight, STYLE_CONFIG.button.borderRadius);
    });
    btnContainer.on('pointerout', () => {
      textObj.setColor(textColor);
      bgObj.clear();
      bgObj.fillStyle(bgColor, 1);
      bgObj.fillRoundedRect(-bgWidth / 2, -bgHeight / 2, bgWidth, bgHeight, STYLE_CONFIG.button.borderRadius);
    });
    btnContainer.on('pointerdown', onClick);

    return btnContainer;
  }

  /**
   * 绘制排行榜板块（响应式布局，文字居中）
   * @param {object} allRecords - 所有排行榜数据
   * @param {number} W - 屏幕宽度
   * @param {number} H - 屏幕高度
   */
  drawSections(allRecords, W, H) {
    const startY = 120; // 板块起始Y坐标
    const sectionWidth = Math.max(
      parseInt(STYLE_CONFIG.section.width) / 100 * W,
      STYLE_CONFIG.section.minWidth
    ); // 板块宽度（响应式，不小于最小宽度）
    const colGap = W - 2 * sectionWidth - 40; // 两列之间的间距（左右留20px边距）
    const rowGap = 240; // 两行之间的间距（避免挤压）

    // 遍历板块配置，按2列布局渲染
    this.sectionsConfig.forEach((section, index) => {
      const col = index % 2; // 列索引（0：左列，1：右列）
      const row = Math.floor(index / 2); // 行索引（0：第一行，1：第二行）
      // 板块X坐标：左列从20px开始，右列从“左列宽+间距”开始，确保居中
      const sectionX = 20 + col * (sectionWidth + colGap);
      // 板块Y坐标：按行索引累加，确保垂直居中
      const sectionY = startY + row * rowGap;

      // 绘制单个板块
      this.drawSection(sectionX, sectionY, sectionWidth, section.title, section.color, allRecords[section.key]);
    });
  }

  /**
   * 绘制单个排行榜板块（文字对齐，防遮挡）
   * @param {number} x - 板块X坐标
   * @param {number} y - 板块Y坐标
   * @param {number} width - 板块宽度
   * @param {string} title - 板块标题
   * @param {string} titleColor - 标题颜色
   * @param {array} records - 板块对应的记录列表
   */
  drawSection(x, y, width, title, titleColor, records) {
    const height = STYLE_CONFIG.section.height;
    const padding = STYLE_CONFIG.section.padding;

    // 1. 板块背景（带边框，增强层次感）
    const sectionBg = this.add.graphics().setDepth(this.uiDepth);
    sectionBg.fillStyle(STYLE_CONFIG.section.bg, 0.95);
    sectionBg.fillRoundedRect(x, y, width, height, 12);
    sectionBg.lineStyle(2, STYLE_CONFIG.section.border, 1);
    sectionBg.strokeRoundedRect(x, y, width, height, 12);

    // 2. 板块标题（左上角，文字居中对齐）
    this.add.text(x + padding, y + padding, title, {
      fontSize: '20px',
      color: titleColor,
      fontStyle: 'bold',
      fontFamily: STYLE_CONFIG.baseFont,
      shadow: STYLE_CONFIG.textShadow
    }).setOrigin(0, 0.5).setDepth(this.uiDepth + 1);

    // 3. 空记录处理（文字居中显示，避免遮挡）
    if (!records || records.length === 0) {
      this.add.text(x + width / 2, y + height / 2, '暂无记录', {
        fontSize: '18px',
        color: STYLE_CONFIG.color.emptyText,
        fontFamily: STYLE_CONFIG.baseFont,
        shadow: STYLE_CONFIG.textShadow
      }).setOrigin(0.5).setDepth(this.uiDepth + 1);
      return;
    }

    // 4. 记录列表（最多显示3条，文字对齐，防重叠）
    const recordStartY = y + padding * 2 + 10; // 记录起始Y坐标（标题下方）
    const recordGap = 36; // 记录之间的垂直间距
    // 截取前3条记录（避免超出板块高度）
    records.slice(0, 3).forEach((record, index) => {
      const recordY = recordStartY + index * recordGap;

      // 4.1 排名（左对齐，固定宽度，避免文字偏移）
      this.add.text(x + padding * 2, recordY, `第 ${index + 1} 名`, {
        fontSize: '18px',
        color: STYLE_CONFIG.color.rankText,
        fontFamily: STYLE_CONFIG.baseFont,
        shadow: STYLE_CONFIG.textShadow
      }).setOrigin(0, 0.5).setDepth(this.uiDepth + 1);

      // 4.2 图片信息（中间对齐，自动换行，避免超出板块）
      const imageInfo = record.imageName || record.imageKey;
      const difficulty = record.rows && record.cols ? `（${record.rows}x${record.cols}）` : '';
      const fullInfo = `${imageInfo}${difficulty}`;
      // 计算信息文字最大宽度（避免超出中间区域）
      const infoMaxWidth = width - 160; // 左右预留排名和时间的宽度
      this.add.text(x + padding * 6, recordY, fullInfo, {
        fontSize: '22px',
        color: STYLE_CONFIG.color.infoText,
        fontFamily: STYLE_CONFIG.baseFont,
        shadow: STYLE_CONFIG.textShadow,
        wordWrap: { width: infoMaxWidth } // 自动换行，避免文字截断
      }).setOrigin(0, 0.5).setDepth(this.uiDepth + 1);

      // 4.3 耗时（右对齐，固定靠右，避免文字重叠）
      this.add.text(x + width - padding * 2, recordY, `${record.time.toFixed(2)} 秒`, {
        fontSize: '18px',
        color: STYLE_CONFIG.color.timeText,
        fontFamily: STYLE_CONFIG.baseFont,
        shadow: STYLE_CONFIG.textShadow
      }).setOrigin(1, 0.5).setDepth(this.uiDepth + 1);
    });
  }

  /**
   * 导出排行榜记录（优化文字提示）
   */
  exportLeaderboard() {
    try {
      const jsonData = LeaderboardStore.exportJSON(true);
      if (!jsonData) {
        this.showToast('导出失败：暂无排行榜数据');
        return;
      }

      // 创建JSON文件并下载
      const blob = new Blob([jsonData], { type: 'application/json;charset=utf-8' });
      const timestamp = this.formatTimestamp(new Date());
      const downloadLink = document.createElement('a');
      downloadLink.href = URL.createObjectURL(blob);
      downloadLink.download = `puzzle_leaderboard_${timestamp}.json`;
      document.body.appendChild(downloadLink);
      downloadLink.click();
      downloadLink.remove();

      this.showToast('排行榜记录导出成功');
    } catch (error) {
      console.error('[exportLeaderboard] 导出异常：', error);
      this.showToast('导出失败：发生未知错误');
    }
  }

  /**
   * 导入排行榜记录（优化文字提示）
   */
  importLeaderboard() {
    // 创建隐藏的文件选择器
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = 'application/json';
    fileInput.style.display = 'none';
    document.body.appendChild(fileInput);

    // 监听文件选择事件
    fileInput.addEventListener('change', async (e) => {
      const selectedFile = e.target.files?.[0];
      fileInput.remove(); // 移除DOM元素，避免内存泄漏

      if (!selectedFile) return;

      try {
        const fileContent = await selectedFile.text();
        const importResult = LeaderboardStore.importJSON(fileContent, { strategy: 'merge' });

        if (!importResult.ok) {
          this.showToast(`导入失败：${importResult.error || '无效的排行榜文件'}`);
          return;
        }

        // 提示导入结果，并重启场景刷新数据
        const toastText = importResult.strategy === 'replace' 
          ? '导入完成（已替换原有记录）' 
          : '导入完成（已合并至原有记录）';
        this.showToast(toastText);
        this.time.delayedCall(500, () => this.scene.restart());
      } catch (error) {
        console.error('[importLeaderboard] 导入异常：', error);
        this.showToast('导入失败：文件读取错误或格式异常');
      }
    });

    // 触发文件选择
    fileInput.click();
  }

  /**
   * 格式化时间戳（用于导出文件名）
   * @param {Date} date - 日期对象
   * @returns {string} 格式化后的时间戳（YYYYMMDD_HHMMSS）
   */
  formatTimestamp(date) {
    const pad = (num) => String(num).padStart(2, '0');
    return `${date.getFullYear()}${pad(date.getMonth() + 1)}${pad(date.getDate())}_${pad(date.getHours())}${pad(date.getMinutes())}${pad(date.getSeconds())}`;
  }

  /**
   * 显示美化的Toast提示（居中底部，防遮挡，带动画）
   * @param {string} text - 提示内容
   * @param {number} duration - 显示时长（默认1800ms）
   */
  showToast(text, duration = 1800) {
    const W = this.cameras.main.width;
    const H = this.cameras.main.height;
    const centerX = W / 2;
    const toastY = H - 60; // 底部居中位置（避免遮挡排行榜）

    // 销毁之前的Toast（避免重叠）
    if (this._toastNodes) {
      this._toastNodes.forEach(node => node.destroy());
      this._toastNodes = null;
    }

    // 1. 计算Toast背景宽度（根据文字长度自适应）
    const tempText = this.add.text(0, 0, text, {
      fontSize: '18px',
      fontFamily: STYLE_CONFIG.baseFont
    });
    const toastWidth = tempText.width + 40; // 左右内边距各20px
    const toastHeight = 46;
    tempText.destroy();

    // 2. Toast背景（圆角，半透明，带阴影）
    const toastBg = this.add.graphics().setDepth(this.uiDepth + 20);
    toastBg.fillStyle(0x000000, 0.7);
    toastBg.fillRoundedRect(centerX - toastWidth / 2, toastY - toastHeight / 2, toastWidth, toastHeight, 10);

    // 3. Toast文字（居中显示，带阴影）
    const toastText = this.add.text(centerX, toastY, text, {
      fontSize: '18px',
      color: '#ffffff',
      fontFamily: STYLE_CONFIG.baseFont,
      shadow: STYLE_CONFIG.textShadow
    }).setOrigin(0.5).setDepth(this.uiDepth + 21);

    // 存储Toast元素，用于后续销毁
    this._toastNodes = [toastBg, toastText];

    // 4. 自动销毁（淡入淡出动画，提升体验）
    this.time.delayedCall(duration - 300, () => {
      this.tweens.add({
        targets: [toastBg, toastText],
        alpha: 0,
        duration: 300,
        ease: 'Power2',
        onComplete: () => {
          if (this._toastNodes) {
            this._toastNodes.forEach(node => node.destroy());
            this._toastNodes = null;
          }
        }
      });
    });
  }
}