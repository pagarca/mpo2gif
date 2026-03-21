class MPOParser {
    static parse(arrayBuffer) {
        const data = new Uint8Array(arrayBuffer);
        const boundaries = [];

        let pos = 0;
        while (pos < data.length - 1) {
            if (data[pos] !== 0xFF || data[pos + 1] !== 0xD8) break;

            boundaries.push(pos);
            pos = MPOParser._skipFrame(data, pos + 2);
            if (pos < 0) break;

            while (pos < data.length - 1) {
                if (data[pos] === 0xFF && data[pos + 1] === 0xD8) break;
                pos++;
            }
        }

        if (boundaries.length < 2) return [];

        const frameInfos = [];
        for (let i = 0; i < boundaries.length; i++) {
            const start = boundaries[i];
            const end = (i < boundaries.length - 1) ? boundaries[i + 1] : data.length;
            frameInfos.push({ start, end, size: end - start });
        }

        frameInfos.sort((a, b) => b.size - a.size);
        const largestTwo = frameInfos.slice(0, 2);
        largestTwo.sort((a, b) => a.start - b.start);

        const frames = [];
        for (const info of largestTwo) {
            const frameData = data.slice(info.start, info.end);
            frames.push(new Blob([frameData], { type: 'image/jpeg' }));
        }

        return frames;
    }

    static _skipFrame(data, pos) {
        while (pos < data.length - 1) {
            if (data[pos] !== 0xFF) return pos;

            const marker = data[pos + 1];

            if (marker === 0xD8 || marker === 0x00) {
                pos++;
                continue;
            }

            if (marker === 0xD9) return pos + 2;

            if (marker === 0xDA) {
                pos += 2;
                const segLen = (data[pos] << 8) | data[pos + 1];
                pos += segLen;
                return MPOParser._skipEntropy(data, pos);
            }

            if (marker >= 0xD0 && marker <= 0xD7) {
                pos += 2;
                continue;
            }

            pos += 2;
            if (pos + 1 < data.length) {
                const segLen = (data[pos] << 8) | data[pos + 1];
                pos += segLen;
            }
        }

        return -1;
    }

    static _skipEntropy(data, pos) {
        while (pos < data.length - 1) {
            if (data[pos] !== 0xFF) {
                pos++;
                continue;
            }

            const next = data[pos + 1];
            if (next === 0x00) {
                pos += 2;
            } else if (next === 0xD9) {
                return pos + 2;
            } else if (next >= 0xD0 && next <= 0xD7) {
                pos += 2;
            } else {
                return pos;
            }
        }

        return -1;
    }
}

class StereoProcessor {
    constructor() {
        this.cvReady = false;
    }

    async waitForOpenCv(timeout = 60000) {
        if (this.cvReady) return true;

        await new Promise((resolve, reject) => {
            const deadline = Date.now() + timeout;
            const check = () => {
                if (window.cv && cv.Mat) {
                    this.cvReady = true;
                    resolve();
                } else if (Date.now() > deadline) {
                    reject(new Error('OpenCV.js load timeout'));
                } else {
                    setTimeout(check, 100);
                }
            };
            check();
        });
        return true;
    }

    async findMatchingPoint(leftCanvas, rightCanvas, point, windowSize = 40, searchRadius = 150) {
        await this.waitForOpenCv();
        
        const cv = window.cv;
        
        const leftMat = cv.imread(leftCanvas);
        const rightMat = cv.imread(rightCanvas);
        
        const leftGray = new cv.Mat();
        const rightGray = new cv.Mat();
        cv.cvtColor(leftMat, leftGray, cv.COLOR_RGBA2GRAY);
        cv.cvtColor(rightMat, rightGray, cv.COLOR_RGBA2GRAY);
        
        const [px, py] = [Math.round(point.x), Math.round(point.y)];
        const w = leftGray.cols;
        const h = leftGray.rows;
        
        const x1 = Math.max(0, px - windowSize);
        const y1 = Math.max(0, py - windowSize);
        const x2 = Math.min(w, px + windowSize);
        const y2 = Math.min(h, py + windowSize);
        
        if (x2 <= x1 || y2 <= y1) {
            leftMat.delete();
            rightMat.delete();
            leftGray.delete();
            rightGray.delete();
            return point;
        }
        
        const template = leftGray.roi(new cv.Rect(x1, y1, x2 - x1, y2 - y1));
        
        const sx1 = Math.max(0, px - searchRadius);
        const sy1 = Math.max(0, py - searchRadius);
        const sx2 = Math.min(w, px + searchRadius);
        const sy2 = Math.min(h, py + searchRadius);
        
        if (sx2 <= sx1 || sy2 <= sy1 || sx2 - sx1 < x2 - x1 || sy2 - sy1 < y2 - y1) {
            template.delete();
            leftMat.delete();
            rightMat.delete();
            leftGray.delete();
            rightGray.delete();
            return point;
        }
        
        const searchArea = rightGray.roi(new cv.Rect(sx1, sy1, sx2 - sx1, sy2 - sy1));
        const result = new cv.Mat();
        cv.matchTemplate(searchArea, template, result, cv.TM_CCOEFF_NORMED);
        
        const minMax = cv.minMaxLoc(result);
        const maxVal = minMax.maxVal;
        const maxLoc = minMax.maxLoc;
        
        template.delete();
        searchArea.delete();
        result.delete();
        leftMat.delete();
        rightMat.delete();
        leftGray.delete();
        rightGray.delete();
        
        if (maxVal < 0.3) {
            return point;
        }
        
        const matchX = maxLoc.x + sx1 + Math.floor((x2 - x1) / 2);
        const matchY = maxLoc.y + sy1 + Math.floor((y2 - y1) / 2);
        
        return { x: matchX, y: matchY };
    }

    async findOptimalCrop(leftCanvas, rightCanvas) {
        await this.waitForOpenCv();
        
        const cv = window.cv;
        
        const leftMat = cv.imread(leftCanvas);
        const rightMat = cv.imread(rightCanvas);
        
        const leftGray = new cv.Mat();
        const rightGray = new cv.Mat();
        cv.cvtColor(leftMat, leftGray, cv.COLOR_RGBA2GRAY);
        cv.cvtColor(rightMat, rightGray, cv.COLOR_RGBA2GRAY);
        
        const orb = new cv.ORB(2000);
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
        
        const bf = new cv.DescriptorMatcher('BruteForce-Hamming');
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

class ImageProcessor {
    static shiftImage(imgData, shiftX, shiftY) {
        const canvas = document.createElement('canvas');
        canvas.width = imgData.width;
        canvas.height = imgData.height;
        const ctx = canvas.getContext('2d');
        
        ctx.drawImage(imgData, 0, 0);
        
        const srcData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const dstData = ctx.createImageData(canvas.width, canvas.height);
        
        for (let y = 0; y < canvas.height; y++) {
            for (let x = 0; x < canvas.width; x++) {
                const srcX = Math.max(0, Math.min(canvas.width - 1, x - shiftX));
                const srcY = Math.max(0, Math.min(canvas.height - 1, y - shiftY));
                
                const dstIdx = (y * canvas.width + x) * 4;
                const srcIdx = (srcY * canvas.width + srcX) * 4;
                
                dstData.data[dstIdx] = srcData.data[srcIdx];
                dstData.data[dstIdx + 1] = srcData.data[srcIdx + 1];
                dstData.data[dstIdx + 2] = srcData.data[srcIdx + 2];
                dstData.data[dstIdx + 3] = srcData.data[srcIdx + 3];
            }
        }
        
        ctx.putImageData(dstData, 0, 0);
        return canvas;
    }
    
    static rotateCanvas(canvas, angle) {
        if (angle === 0) {
            const copy = document.createElement('canvas');
            copy.width = canvas.width;
            copy.height = canvas.height;
            copy.getContext('2d').drawImage(canvas, 0, 0);
            return copy;
        }
        
        const radians = angle * Math.PI / 180;
        const sin = Math.abs(Math.sin(radians));
        const cos = Math.abs(Math.cos(radians));
        
        const newWidth = Math.round(canvas.width * cos + canvas.height * sin);
        const newHeight = Math.round(canvas.width * sin + canvas.height * cos);
        
        const rotated = document.createElement('canvas');
        rotated.width = newWidth;
        rotated.height = newHeight;
        
        const ctx = rotated.getContext('2d');
        ctx.translate(newWidth / 2, newHeight / 2);
        ctx.rotate(radians);
        ctx.drawImage(canvas, -canvas.width / 2, -canvas.height / 2);
        
        return rotated;
    }
    
    static cropBorders(canvas, cropPx) {
        if (cropPx <= 0) {
            const copy = document.createElement('canvas');
            copy.width = canvas.width;
            copy.height = canvas.height;
            copy.getContext('2d').drawImage(canvas, 0, 0);
            return copy;
        }
        
        const newWidth = canvas.width - 2 * cropPx;
        const newHeight = canvas.height - 2 * cropPx;
        
        if (newWidth <= 0 || newHeight <= 0) {
            return canvas;
        }
        
        const cropped = document.createElement('canvas');
        cropped.width = newWidth;
        cropped.height = newHeight;
        
        cropped.getContext('2d').drawImage(
            canvas, cropPx, cropPx, newWidth, newHeight,
            0, 0, newWidth, newHeight
        );
        
        return cropped;
    }
}

class GIFGenerator {
    static async generateFromFrames(leftCanvas, rightCanvas, duration) {
        return new Promise((resolve, reject) => {
            const gif = new GIF({
                workers: 2,
                workerScript: 'gif.worker.js',
                quality: 10,
            });
            
            const ctx1 = leftCanvas.getContext('2d');
            const imageData1 = ctx1.getImageData(0, 0, leftCanvas.width, leftCanvas.height);
            gif.addFrame(imageData1, { delay: duration, copy: true });
            
            const ctx2 = rightCanvas.getContext('2d');
            const imageData2 = ctx2.getImageData(0, 0, rightCanvas.width, rightCanvas.height);
            gif.addFrame(imageData2, { delay: duration, copy: true });
            
            gif.on('finished', blob => resolve(blob));
            gif.on('error', reject);
            gif.render();
        });
    }
}

class MP4Generator {
    static async generate(frames, duration, lengthSeconds = 5) {
        const fps = Math.round(1000 / duration);
        const totalFrames = fps * lengthSeconds;
        
        const canvasFrames = [];
        
        for (let i = 0; i < totalFrames; i++) {
            const isLeft = i % 2 === 0;
            const srcCanvas = isLeft ? frames.left : frames.right;
            
            const frameCanvas = document.createElement('canvas');
            frameCanvas.width = srcCanvas.width;
            frameCanvas.height = srcCanvas.height;
            frameCanvas.getContext('2d').drawImage(srcCanvas, 0, 0);
            
            canvasFrames.push(frameCanvas);
        }
        
        return new Promise((resolve, reject) => {
            const stream = canvasFrames[0].captureStream(fps);
            const mediaRecorder = new MediaRecorder(stream, {
                mimeType: 'video/webm;codecs=vp9',
                videoBitsPerSecond: 5000000
            });
            
            const chunks = [];
            mediaRecorder.ondataavailable = e => chunks.push(e.data);
            mediaRecorder.onstop = () => {
                const blob = new Blob(chunks, { type: 'video/webm' });
                resolve(blob);
            };
            mediaRecorder.onerror = reject;
            
            mediaRecorder.start();
            
            let frameIndex = 0;
            const frameInterval = setInterval(() => {
                if (frameIndex >= canvasFrames.length) {
                    clearInterval(frameInterval);
                    mediaRecorder.stop();
                    return;
                }
                
                stream.getVideoTracks()[0].requestFrame();
                frameIndex++;
            }, 1000 / fps);
        });
    }
}

class App {
    constructor() {
        this.leftOrig = null;
        this.rightOrig = null;
        this.leftImg = null;
        this.rightImg = null;
        this.leftAligned = null;
        this.rightAligned = null;
        this.rotation = 0;
        this.focalLeft = null;
        this.focalRight = null;
        this.halfDx = 0;
        this.halfDy = 0;
        this.duration = 150;
        this.processor = new StereoProcessor();
        this.previewAnimId = null;
        this.previewFrame = 0;
        this.previewLastSwitch = 0;
        this.isProcessing = false;
        
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
        this.focalInfo = document.getElementById('focal-info');
        this.previewStatus = document.getElementById('preview-status');
        this.durationSlider = document.getElementById('duration-slider');
        this.durationValue = document.getElementById('duration-value');
        this.generateBtn = document.getElementById('generate-btn');
        this.generateMp4Btn = document.getElementById('generate-mp4-btn');
        this.generating = document.getElementById('generating');
        this.generatingStatus = document.getElementById('generating-status');
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
        
        this.leftCanvas.addEventListener('click', e => this.onLeftClick(e));
        
        this.durationSlider.addEventListener('input', e => {
            this.duration = parseInt(e.target.value);
            this.durationValue.textContent = this.duration;
            this.resetGenerateButtons();
        });
        
        document.getElementById('rot-cw').addEventListener('click', () => this.rotate(270));
        document.getElementById('rot-ccw').addEventListener('click', () => this.rotate(90));
        document.getElementById('rot-180').addEventListener('click', () => this.rotate(180));
        document.getElementById('rot-reset').addEventListener('click', () => this.rotate(0));
        
        this.generateBtn.addEventListener('click', () => this.generateGIF());
        this.generateMp4Btn.addEventListener('click', () => this.generateMP4());
    }

    async handleFile(file) {
        if (!file.name.toLowerCase().endsWith('.mpo')) {
            alert('Please upload a valid .MPO file');
            return;
        }
        
        this.resetState();
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
            
            this.updateDisplay();
            
            this.hideLoading();
            this.workspace.classList.remove('hidden');
            
        } catch (error) {
            console.error(error);
            alert('Error processing file: ' + error.message);
            this.hideLoading();
        }
    }

    resetState() {
        this.leftOrig = null;
        this.rightOrig = null;
        this.leftImg = null;
        this.rightImg = null;
        this.leftAligned = null;
        this.rightAligned = null;
        this.rotation = 0;
        this.focalLeft = null;
        this.focalRight = null;
        this.halfDx = 0;
        this.halfDy = 0;
        this.previewFrame = 0;
        this.previewLastSwitch = 0;
        if (this.previewAnimId) {
            cancelAnimationFrame(this.previewAnimId);
            this.previewAnimId = null;
        }
    }

    async loadFrames(frames) {
        return new Promise((resolve, reject) => {
            const leftImg = new Image();
            const rightImg = new Image();
            
            let loaded = 0;
            const checkLoaded = () => {
                loaded++;
                if (loaded === 2) resolve();
            };
            
            leftImg.onload = () => {
                this.leftOrig = leftImg;
                this.leftImg = leftImg;
                this.leftCanvas.width = leftImg.naturalWidth || leftImg.width;
                this.leftCanvas.height = leftImg.naturalHeight || leftImg.height;
                const ctx = this.leftCanvas.getContext('2d');
                ctx.drawImage(leftImg, 0, 0);
                checkLoaded();
            };
            
            rightImg.onload = () => {
                this.rightOrig = rightImg;
                this.rightImg = rightImg;
                this.rightCanvas.width = rightImg.naturalWidth || rightImg.width;
                this.rightCanvas.height = rightImg.naturalHeight || rightImg.height;
                const ctx = this.rightCanvas.getContext('2d');
                ctx.drawImage(rightImg, 0, 0);
                checkLoaded();
            };
            
            leftImg.onerror = rightImg.onerror = reject;
            
            leftImg.src = URL.createObjectURL(frames[0]);
            rightImg.src = URL.createObjectURL(frames[1]);
        });
    }

    rotate(angle) {
        if (!this.leftOrig || !this.rightOrig) return;
        
        if (angle === 0) {
            this.rotation = 0;
        } else {
            this.rotation = (this.rotation + angle) % 360;
        }
        
        this.leftImg = ImageProcessor.rotateCanvas(this.leftOrig, this.rotation);
        this.rightImg = ImageProcessor.rotateCanvas(this.rightOrig, this.rotation);
        
        this.leftCanvas.width = this.leftImg.width;
        this.leftCanvas.height = this.leftImg.height;
        this.rightCanvas.width = this.rightImg.width;
        this.rightCanvas.height = this.rightImg.height;
        
        this.focalLeft = null;
        this.focalRight = null;
        this.leftAligned = null;
        this.rightAligned = null;
        
        this.focalInfo.textContent = 'Click on the left image to set the focal point';
        this.updateDisplay();
    }

    async onLeftClick(event) {
        if (!this.leftImg || !this.rightImg || this.isProcessing) return;
        
        const rect = this.leftCanvas.getBoundingClientRect();
        const canvasX = event.clientX - rect.left;
        const canvasY = event.clientY - rect.top;
        
        const cw = this.leftCanvas.width;
        const ch = this.leftCanvas.height;
        
        if (cw <= 0 || ch <= 0) return;
        
        const cssScaleX = rect.width / cw;
        const cssScaleY = rect.height / ch;
        
        const imgX = canvasX / cssScaleX;
        const imgY = canvasY / cssScaleY;
        
        if (imgX < 0 || imgX >= cw || imgY < 0 || imgY >= ch) return;
        
        this.isProcessing = true;
        this.focalInfo.textContent = `Finding match for (${Math.round(imgX)}, ${Math.round(imgY)})...`;
        
        try {
            const focalLeft = { x: Math.round(imgX), y: Math.round(imgY) };
            
            const focalRight = await this.processor.findMatchingPoint(
                this.leftCanvas, this.rightCanvas, focalLeft
            );
            
            this.focalLeft = focalLeft;
            this.focalRight = focalRight;
            
            const dx = this.focalLeft.x - this.focalRight.x;
            const dy = this.focalLeft.y - this.focalRight.y;
            
            this.halfDx = Math.round(dx / 2);
            this.halfDy = Math.round(dy / 2);
            
            const leftShifted = ImageProcessor.shiftImage(this.leftImg, -this.halfDx, -this.halfDy);
            const rightShifted = ImageProcessor.shiftImage(this.rightImg, this.halfDx, this.halfDy);
            
            const cropPx = Math.max(Math.abs(this.halfDx), Math.abs(this.halfDy));
            this.leftAligned = ImageProcessor.cropBorders(leftShifted, cropPx);
            this.rightAligned = ImageProcessor.cropBorders(rightShifted, cropPx);
            
            this.focalInfo.textContent = 
                `Focal: (${this.focalLeft.x}, ${this.focalLeft.y}) → (${this.focalRight.x}, ${this.focalRight.y}) | Disparity: dx=${dx}, dy=${dy}`;
            
            this.updateDisplay();
            
        } catch (error) {
            console.error(error);
            this.focalInfo.textContent = 'Error finding match. Click again.';
        }
        
        this.isProcessing = false;
    }

    updateDisplay() {
        if (!this.leftImg || !this.rightImg) return;
        
        this.drawCanvasWithMarker(this.leftCanvas, this.leftImg, this.focalLeft);
        this.drawCanvasWithMarker(this.rightCanvas, this.rightImg, this.focalRight);
        this.updatePreview();
    }

    drawCanvasWithMarker(canvas, img, markerPt) {
        const cw = canvas.width;
        const ch = canvas.height;
        const iw = img.width;
        const ih = img.height;
        
        if (cw <= 1 || ch <= 1 || iw <= 0 || ih <= 0) return;
        
        const mainCtx = canvas.getContext('2d');
        mainCtx.clearRect(0, 0, cw, ch);
        mainCtx.drawImage(img, 0, 0);
        
        if (markerPt) {
            const mx = markerPt.x;
            const my = markerPt.y;
            const r = 12;
            
            mainCtx.strokeStyle = '#FF0000';
            mainCtx.lineWidth = 3;
            
            mainCtx.beginPath();
            mainCtx.moveTo(mx - r, my);
            mainCtx.lineTo(mx + r, my);
            mainCtx.stroke();
            
            mainCtx.beginPath();
            mainCtx.moveTo(mx, my - r);
            mainCtx.lineTo(mx, my + r);
            mainCtx.stroke();
            
            mainCtx.beginPath();
            mainCtx.ellipse(mx, my, r / 2, r / 2, 0, 0, Math.PI * 2);
            mainCtx.stroke();
        }
    }

    updatePreview() {
        if (this.previewAnimId) {
            cancelAnimationFrame(this.previewAnimId);
            this.previewAnimId = null;
        }
        
        let leftFrame, rightFrame;
        
        if (this.focalLeft && this.leftAligned && this.rightAligned) {
            leftFrame = this.leftAligned;
            rightFrame = this.rightAligned;
            this.previewStatus.textContent = '(focal point aligned)';
        } else {
            leftFrame = this.leftImg;
            rightFrame = this.rightImg;
            this.previewStatus.textContent = '';
        }
        
        if (!leftFrame || !rightFrame) return;
        
        const w = leftFrame.width;
        const h = leftFrame.height;
        
        if (w <= 0 || h <= 0) return;
        
        this.previewCanvas.width = w;
        this.previewCanvas.height = h;
        
        const ctx = this.previewCanvas.getContext('2d');
        this.previewFrame = 0;
        this.previewLastSwitch = 0;
        
        const drawFrame = (frame) => {
            ctx.clearRect(0, 0, w, h);
            ctx.drawImage(frame, 0, 0);
        };
        
        drawFrame(leftFrame);
        
        const animate = (timestamp) => {
            if (!this.previewLastSwitch || timestamp - this.previewLastSwitch >= this.duration) {
                this.previewFrame = 1 - this.previewFrame;
                this.previewLastSwitch = timestamp;
                
                const frame = this.previewFrame === 0 ? leftFrame : rightFrame;
                drawFrame(frame);
            }
            
            this.previewAnimId = requestAnimationFrame(animate);
        };
        
        this.previewAnimId = requestAnimationFrame(animate);
    }

    getOutputFrames() {
        if (this.focalLeft && this.leftAligned && this.rightAligned) {
            return { left: this.leftAligned, right: this.rightAligned };
        }
        
        return { left: this.leftImg, right: this.rightImg };
    }

    resetGenerateButtons() {
        this.generateBtn.textContent = 'Generate & Download GIF';
        this.generateMp4Btn.textContent = 'Generate & Download MP4';
    }

    async generateGIF() {
        this.generateBtn.disabled = true;
        this.generateMp4Btn.disabled = true;
        this.generatingStatus.textContent = 'Generating GIF...';
        this.generating.classList.remove('hidden');
        
        try {
            const frames = this.getOutputFrames();
            const blob = await GIFGenerator.generateFromFrames(frames.left, frames.right, this.duration);
            
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'wiggle.gif';
            a.click();
            URL.revokeObjectURL(url);
            
            this.generateBtn.textContent = 'GIF Downloaded!';
        } catch (error) {
            console.error(error);
            alert('Error generating GIF: ' + error.message);
            this.resetGenerateButtons();
        } finally {
            this.generating.classList.add('hidden');
            this.generateBtn.disabled = false;
            this.generateMp4Btn.disabled = false;
        }
    }

    async generateMP4() {
        this.generateBtn.disabled = true;
        this.generateMp4Btn.disabled = true;
        this.generatingStatus.textContent = 'Generating MP4...';
        this.generating.classList.remove('hidden');
        
        try {
            const frames = this.getOutputFrames();
            const blob = await MP4Generator.generate(frames, this.duration, 5);
            
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'wiggle.webm';
            a.click();
            URL.revokeObjectURL(url);
            
            this.generateMp4Btn.textContent = 'MP4 Downloaded!';
        } catch (error) {
            console.error(error);
            alert('Error generating MP4: ' + error.message);
            this.resetGenerateButtons();
        } finally {
            this.generating.classList.add('hidden');
            this.generateBtn.disabled = false;
            this.generateMp4Btn.disabled = false;
        }
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
