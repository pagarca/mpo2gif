class MPOParser {
    static parse(arrayBuffer) {
        const data = new Uint8Array(arrayBuffer);
        const frames = [];
        
        let pos = 0;
        const frameStarts = [];
        
        while (pos < data.length - 1) {
            if (data[pos] === 0xFF && data[pos + 1] === 0xD8) {
                frameStarts.push(pos);
            }
            pos++;
        }
        
        if (frameStarts.length < 2) {
            return frames;
        }
        
        const frameInfos = [];
        for (let i = 0; i < frameStarts.length; i++) {
            const start = frameStarts[i];
            const end = (i < frameStarts.length - 1) ? frameStarts[i + 1] : data.length;
            const size = end - start;
            frameInfos.push({ start, end, size, index: i });
        }
        
        frameInfos.sort((a, b) => b.size - a.size);
        
        const largestTwo = frameInfos.slice(0, 2);
        largestTwo.sort((a, b) => a.start - b.start);
        
        for (const info of largestTwo) {
            const frameData = data.slice(info.start, info.end);
            frames.push(new Blob([frameData], { type: 'image/jpeg' }));
        }
        
        return frames;
    }
}

class StereoProcessor {
    constructor() {
        this.cvReady = false;
    }

    async waitForOpenCV(timeout = 60000) {
        if (this.cvReady) return true;
        
        const startTime = Date.now();
        while (!window.cv || !window.cv.ORB_create) {
            if (Date.now() - startTime > timeout) {
                throw new Error('OpenCV.js load timeout');
            }
            await new Promise(resolve => setTimeout(resolve, 200));
        }
        this.cvReady = true;
        return true;
    }

    async findOptimalCrop(leftImg, rightImg) {
        await this.waitForOpenCV();
        
        const cv = window.cv;
        
        const leftMat = cv.imread(leftImg);
        const rightMat = cv.imread(rightImg);
        
        const leftGray = new cv.Mat();
        const rightGray = new cv.Mat();
        cv.cvtColor(leftMat, leftGray, cv.COLOR_RGBA2GRAY);
        cv.cvtColor(rightMat, rightGray, cv.COLOR_RGBA2GRAY);
        
        const orb = cv.ORB_create(2000);
        const kp1 = new cv.KeyPointVector();
        const kp2 = new cv.KeyPointVector();
        const des1 = new cv.Mat();
        const des2 = new cv.Mat();
        
        orb.detectAndCompute(leftGray, new cv.Mat(), kp1, des1);
        orb.detectAndCompute(rightGray, new cv.Mat(), kp2, des2);
        
        if (des1.rows < 10 || des2.rows < 10) {
            leftMat.delete();
            rightMat.delete();
            leftGray.delete();
            rightGray.delete();
            kp1.delete();
            kp2.delete();
            des1.delete();
            des2.delete();
            return 0;
        }
        
        const bf = new cv.BFMatcher(cv.NORM_HAMMING, true);
        const matches = new cv.DMatchVector();
        bf.match(des1, des2, matches);
        
        if (matches.size() < 10) {
            leftMat.delete();
            rightMat.delete();
            leftGray.delete();
            rightGray.delete();
            kp1.delete();
            kp2.delete();
            des1.delete();
            des2.delete();
            bf.delete();
            matches.delete();
            return 0;
        }
        
        const disparities = [];
        for (let i = 0; i < matches.size(); i++) {
            const m = matches.get(i);
            const dx = kp1.get(m.queryIdx).pt.x - kp2.get(m.trainIdx).pt.x;
            disparities.push(dx);
        }
        
        disparities.sort((a, b) => a - b);
        const medianIdx = Math.floor(disparities.length / 2);
        const medianDisp = Math.abs(disparities[medianIdx]);
        const cropValue = Math.round(medianDisp / 5) * 5;
        
        leftMat.delete();
        rightMat.delete();
        leftGray.delete();
        rightGray.delete();
        kp1.delete();
        kp2.delete();
        des1.delete();
        des2.delete();
        bf.delete();
        matches.delete();
        
        return Math.max(0, cropValue);
    }
}

class GIFGenerator {
    static async generate(leftCanvas, rightCanvas, duration) {
        return new Promise((resolve, reject) => {
            const gif = new GIF({
                workers: 2,
                workerScript: 'https://cdnjs.cloudflare.com/ajax/libs/gif.js/0.2.0/gif.worker.js',
                quality: 10,
            });
            
            const ctx = leftCanvas.getContext('2d');
            const imageData = ctx.getImageData(0, 0, leftCanvas.width, leftCanvas.height);
            gif.addFrame(imageData, { delay: duration });
            
            const ctx2 = rightCanvas.getContext('2d');
            const imageData2 = ctx2.getImageData(0, 0, rightCanvas.width, rightCanvas.height);
            gif.addFrame(imageData2, { delay: duration });
            
            gif.on('finished', blob => resolve(blob));
            gif.on('error', reject);
            gif.render();
        });
    }
}

class App {
    constructor() {
        this.leftImg = null;
        this.rightImg = null;
        this.autoCrop = 0;
        this.currentCrop = 0;
        this.duration = 150;
        this.generatedBlob = null;
        this.processor = new StereoProcessor();
        
        this.initElements();
        this.bindEvents();
    }

    initElements() {
        this.uploadArea = document.getElementById('upload-area');
        this.fileInput = document.getElementById('file-input');
        this.loadingOpenCV = document.getElementById('loading-opencv');
        this.processing = document.getElementById('processing');
        this.processingStatus = document.getElementById('processing-status');
        this.workspace = document.getElementById('workspace');
        this.leftCanvas = document.getElementById('left-canvas');
        this.rightCanvas = document.getElementById('right-canvas');
        this.previewCanvas = document.getElementById('preview-canvas');
        this.cropSlider = document.getElementById('crop-slider');
        this.cropValue = document.getElementById('crop-value');
        this.autoCropValue = document.getElementById('auto-crop-value');
        this.durationSlider = document.getElementById('duration-slider');
        this.durationValue = document.getElementById('duration-value');
        this.generateBtn = document.getElementById('generate-btn');
        this.downloadBtn = document.getElementById('download-btn');
        this.generating = document.getElementById('generating');
    }

    bindEvents() {
        this.uploadArea.addEventListener('click', () => this.fileInput.click());
        this.uploadArea.addEventListener('dragover', e => {
            e.preventDefault();
            this.uploadArea.classList.add('dragover');
        });
        this.uploadArea.addEventListener('dragleave', () => {
            this.uploadArea.classList.remove('dragover');
        });
        this.uploadArea.addEventListener('drop', e => {
            e.preventDefault();
            this.uploadArea.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            if (file) this.handleFile(file);
        });
        
        this.fileInput.addEventListener('change', e => {
            const file = e.target.files[0];
            if (file) this.handleFile(file);
        });
        
        this.cropSlider.addEventListener('input', e => {
            this.currentCrop = parseInt(e.target.value);
            this.cropValue.textContent = this.currentCrop;
            this.updatePreview();
        });
        
        this.durationSlider.addEventListener('input', e => {
            this.duration = parseInt(e.target.value);
            this.durationValue.textContent = this.duration;
        });
        
        this.generateBtn.addEventListener('click', () => this.generateGIF());
        this.downloadBtn.addEventListener('click', () => this.downloadGIF());
    }

    async handleFile(file) {
        if (!file.name.toLowerCase().endsWith('.mpo')) {
            alert('Please upload a valid .MPO file');
            return;
        }
        
        this.showLoading('Loading...');
        
        try {
            const buffer = await file.arrayBuffer();
            const frames = MPOParser.parse(buffer);
            
            if (frames.length < 2) {
                alert('This MPO file does not contain stereo frames');
                this.hideLoading();
                return;
            }
            
            this.showLoading('Loading OpenCV.js (first time only)...');
            
            await this.loadFrames(frames);
            
            this.showLoading('Analyzing stereo offset...');
            this.autoCrop = await this.processor.findOptimalCrop(this.leftCanvas, this.rightCanvas);
            this.autoCropValue.textContent = this.autoCrop;
            
            this.currentCrop = this.autoCrop;
            this.cropSlider.value = this.currentCrop;
            this.cropValue.textContent = this.currentCrop;
            
            this.updatePreview();
            
            this.hideLoading();
            this.workspace.classList.remove('hidden');
            this.downloadBtn.disabled = true;
            this.generatedBlob = null;
            
        } catch (error) {
            console.error(error);
            alert('Error processing file: ' + error.message);
            this.hideLoading();
        }
    }

    async loadFrames(frames) {
        return new Promise((resolve, reject) => {
            this.leftImg = new Image();
            this.rightImg = new Image();
            
            let loaded = 0;
            const checkLoaded = () => {
                loaded++;
                if (loaded === 2) resolve();
            };
            
            this.leftImg.onload = () => {
                this.leftCanvas.width = this.leftImg.width;
                this.leftCanvas.height = this.leftImg.height;
                this.leftCanvas.getContext('2d').drawImage(this.leftImg, 0, 0);
                checkLoaded();
            };
            
            this.rightImg.onload = () => {
                this.rightCanvas.width = this.rightImg.width;
                this.rightCanvas.height = this.rightImg.height;
                this.rightCanvas.getContext('2d').drawImage(this.rightImg, 0, 0);
                checkLoaded();
            };
            
            this.leftImg.onerror = this.rightImg.onerror = reject;
            
            this.leftImg.src = URL.createObjectURL(frames[0]);
            this.rightImg.src = URL.createObjectURL(frames[1]);
        });
    }

    updatePreview() {
        const w = this.leftImg.width;
        const h = this.leftImg.height;
        const crop = this.currentCrop;
        
        const croppedW = w - crop;
        
        this.previewCanvas.width = croppedW;
        this.previewCanvas.height = h;
        
        const ctx = this.previewCanvas.getContext('2d');
        
        ctx.drawImage(this.leftImg, 0, 0, croppedW, h, 0, 0, croppedW, h);
    }

    async generateGIF() {
        this.generateBtn.disabled = true;
        this.generating.classList.remove('hidden');
        
        try {
            const w = this.leftImg.width;
            const h = this.leftImg.height;
            const crop = this.currentCrop;
            const croppedW = w - crop;
            
            const tempCanvas1 = document.createElement('canvas');
            const tempCanvas2 = document.createElement('canvas');
            tempCanvas1.width = croppedW;
            tempCanvas1.height = h;
            tempCanvas2.width = croppedW;
            tempCanvas2.height = h;
            
            const ctx1 = tempCanvas1.getContext('2d');
            const ctx2 = tempCanvas2.getContext('2d');
            
            ctx1.drawImage(this.leftImg, 0, 0, croppedW, h, 0, 0, croppedW, h);
            ctx2.drawImage(this.rightImg, crop, 0, croppedW, h, 0, 0, croppedW, h);
            
            this.generatedBlob = await GIFGenerator.generate(tempCanvas1, tempCanvas2, this.duration);
            
            this.downloadBtn.disabled = false;
        } catch (error) {
            console.error(error);
            alert('Error generating GIF: ' + error.message);
        } finally {
            this.generating.classList.add('hidden');
            this.generateBtn.disabled = false;
        }
    }

    downloadGIF() {
        if (!this.generatedBlob) return;
        
        const url = URL.createObjectURL(this.generatedBlob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'wiggle.gif';
        a.click();
        URL.revokeObjectURL(url);
    }

    showLoading(status) {
        this.processingStatus.textContent = status;
        this.processing.classList.remove('hidden');
    }

    hideLoading() {
        this.processing.classList.add('hidden');
        this.loadingOpenCV.classList.add('hidden');
    }
}

let app;
window.onOpenCvReady = function() {
    console.log('OpenCV.js is ready');
};

document.addEventListener('DOMContentLoaded', () => {
    app = new App();
});
