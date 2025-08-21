const { app, BrowserWindow, Tray, Menu, nativeImage } = require('electron');
const path = require('path');

let tray = null;
let mainWindow = null;

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 500,
        height: 700,
        show: false, // Start hidden
        frame: false, // No window frame
        resizable: false,
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'), // Optional: for secure IPC
            nodeIntegration: true,
            contextIsolation: false
        }
    });

    // In development, load from React's server. In production, load from build file.
    mainWindow.loadURL('http://localhost:3000'); 
    
    mainWindow.on('closed', () => mainWindow = null);
    
    // Hide the window when it loses focus
    mainWindow.on('blur', () => {
        if (!mainWindow.webContents.isDevToolsOpened()) {
            mainWindow.hide();
        }
    });
}

function createTray() {
    // Create a native image from a simple data URL or load a file
    const icon = nativeImage.createFromPath(path.join(__dirname, 'public/icon.png')); // Make sure you have an icon.png
    tray = new Tray(icon);

    const contextMenu = Menu.buildFromTemplate([
        { label: 'Show App', click: () => {
            mainWindow.show();
        }},
        { label: 'Quit', click: () => {
            app.isQuitting = true;
            app.quit();
        }}
    ]);

    tray.setToolTip('AI Desktop Agent');
    tray.setContextMenu(contextMenu);

    tray.on('click', () => {
        mainWindow.isVisible() ? mainWindow.hide() : mainWindow.show();
    });
}

app.on('ready', () => {
    createWindow();
    createTray();
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
        createWindow();
    }
});