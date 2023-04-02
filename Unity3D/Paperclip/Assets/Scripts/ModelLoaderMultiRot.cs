using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using Dummiesman;
using System.IO;
using UnityEngine.Assertions;

public class ModelLoaderMultiRot : MonoBehaviour {
    
    List<string> fileList;
    public GameObject cameraObject;
    private GameObject modelObjectX;  // Parent object used for rotation around x-axis
    private GameObject modelObjectY;  // Parent object used for rotation around y-axis
    private GameObject modelObjectZ;  // Parent object used for rotation around z-axis
    private GameObject modelObjectParent;  // Contains the all the other models i.e. the parent object
    private GameObject modelObjectChild;  // Contains the main model which is initially rotated and fixed
    private float maxSizeLimit;
    private float screenshotDelay;
    private int maxScreenshots;
    private int numScreenshots;
    private int frameIterator;
    private int numObjectsGenerated;
    private List<Vector3> sourceVertexList;
    private string screenshotOutputDir;
    private int maxObjects;
    private string rotationAxis;
    private bool iterateOverAxis;
    private List<string> axisList;
    private int axisScreenshots;
    private int multiAxisScreenshots;
    private float axisRotationStride;
    private float multiAxisRotationStride;
    private List<int> axisScreenshotsList;
    private List<int> cummulativeAxisScreenshotsList;
    private int currentAxisIterator;
    private string outputDirPostfix;
    private bool useMaterials;
    private Vector3 currentObjectRot;

    void Start () {
        // Disable extention logging
        Debug.unityLogger.filterLogType = LogType.Exception;

        // Set global variables
        rotationAxis = "y";
        axisList = new List<string>();
        axisList.Add("x");
        axisList.Add("y");
        axisList.Add("z");
        axisList.Add("xy");
        axisList.Add("xz");
        axisList.Add("yz");
        // axisList.Add("xyz");
        axisRotationStride = 1.0f;
        multiAxisRotationStride = 10.0f;
        maxScreenshots = 360;
        multiAxisScreenshots = maxScreenshots / (int) multiAxisRotationStride;
        axisScreenshots = maxScreenshots / (int) axisRotationStride;
        iterateOverAxis = true;
        outputDirPostfix = "";
        Debug.Log("Axis screenshots: " + axisScreenshots.ToString() + " | Multi-axis screenshots: " + multiAxisScreenshots.ToString());

        axisScreenshotsList = new List<int>();
        cummulativeAxisScreenshotsList = new List<int>();
        for (int i = 0; i < axisList.Count; i++) {
            if (axisList[i].Length == 1) {
                axisScreenshotsList.Add(axisScreenshots);
            } else {
                axisScreenshotsList.Add((int) Mathf.Pow(multiAxisScreenshots, axisList[i].Length));
            }
            if (i > 0) {
                cummulativeAxisScreenshotsList.Add(cummulativeAxisScreenshotsList[i-1] + axisScreenshotsList[i]);
            } else {
                cummulativeAxisScreenshotsList.Add(axisScreenshotsList[i]);
            }
            Debug.Log("Number of screenshots at axis " + axisList[i] + " : " + axisScreenshotsList[i].ToString());
            Debug.Log("Cummulative number of screenshots at axis " + axisList[i] + " : " + cummulativeAxisScreenshotsList[i].ToString());
        }

        useMaterials = false;
        maxObjects = -1;  // Max number of paperclip objects to generate -- negative numbers indicate the maximum number of objects found
        numObjectsGenerated = 0;  // Starting idx
        
        int numFramesPerSec = (int) Mathf.Round(1.0f / Time.deltaTime);  // time.deltatime returns time in seconds between frames
        screenshotDelay = 1;  // Take screenshot on every frame
        if (iterateOverAxis) {
            maxScreenshots = cummulativeAxisScreenshotsList[cummulativeAxisScreenshotsList.Count - 1];  // Last index
            Debug.Log("Maximum screenshots selected to be " + maxScreenshots.ToString() + " frames with axis rotation (axis screenshots: " + axisScreenshots.ToString() + ")...");
        } else {
            maxScreenshots = 360;  // Completes about 360 deg rotation in this time
            Debug.Log("Maximum screenshots selected to be " + maxScreenshots.ToString() + " frames...");
        }
        Debug.Log("Screenshot interval selected to be " + screenshotDelay.ToString() + " frames...");

        // Get the base screenshot directory from the screen recorder
        screenshotOutputDir = cameraObject.GetComponent<ScreenRecorder>().getBaseFolder();

        GetAllModelFiles();
        if (maxObjects <= 0) {
            maxObjects = fileList.Count;
            Debug.Log("Max objects set to be: " + maxObjects.ToString());
        }
        Load3DModel();
    }

    void GetAllModelFiles()
    {
        fileList = new List<string>();
        string targetFileName = "model_normalized.obj";
        
        // Chair -> 03001627; Aeroplane -> 02691156
        string rootDir = @"/mnt/sas/Datasets/ShapeNetCore.v2/03001627/";
        DirectoryInfo d = new DirectoryInfo(rootDir);
        DirectoryInfo[] Dirs = d.GetDirectories();
        foreach (DirectoryInfo dir in Dirs )
        {
            DirectoryInfo completeDir = new DirectoryInfo(rootDir + dir.Name + @"/models/");
            FileInfo[] Files = completeDir.GetFiles("*");
            foreach (FileInfo file in Files)
            {
                if (file.Name.Equals(targetFileName)) {
                    string completeFile = completeDir + targetFileName;
                    Debug.Log("Main file found: " + completeFile);
                    fileList.Add(completeFile);
                }
            }
        }
        fileList.Sort();  // Sort file list to ensure deterministic ordering

        Debug.Log("Total number of files found: " + fileList.Count.ToString());
    }

    void Load3DModel()
    {
        // Reset global vars
        frameIterator = 0;
        numScreenshots = 0;
        currentObjectRot = new Vector3(0.0f, 0.0f, 0.0f);

        currentAxisIterator = 0;
        rotationAxis = axisList[currentAxisIterator];
        outputDirPostfix = "/" + rotationAxis;
        Debug.Log("Rotation axis selected to be " + rotationAxis + " axis after " + numScreenshots.ToString() + " screenshots...");

        // Create the primary game object
        if (numObjectsGenerated >= fileList.Count) {
            Debug.Log("No more files in the list. Terminating application...");
            return;
        }

        string filePath = fileList[numObjectsGenerated];
        Debug.Log("Loading model file: " + filePath);
        modelObjectChild = new OBJLoader().Load(filePath);
        modelObjectChild.name = "3D_Object";
        modelObjectChild.transform.localScale = new Vector3(10, 10, 10);

        if (!useMaterials) {
            Debug.Log("Removing model material...");
            foreach (Transform t in modelObjectChild.transform){
                Material[] myMaterials = t.gameObject.GetComponent<Renderer>().materials;
                foreach (Material m_Material in myMaterials) {
                    m_Material.color = Color.white;
                    m_Material.SetTexture("_MainTex", null);  // Remove the albedo map
                    m_Material.SetTexture("_DetailAlbedoMap", null);  // Remove the albedo map
                }
            }
        }

        bool randomObjectRotation = true;
        if (randomObjectRotation) {
            // Rotate the object randomly before rotating the parent
            // modelObjectChild.transform.localEulerAngles = new Vector3(Random.Range(0, 360), Random.Range(0, 360), Random.Range(0, 360));
            modelObjectChild.transform.rotation = Random.rotation;
        }
        else {
            modelObjectChild.transform.eulerAngles = new Vector3(0.0f, 0.0f, 0.0f);
        }

        // Assign the main game object as parent which will be used for rotation
        modelObjectZ = new GameObject("Parent_GO_Z");
        modelObjectZ.transform.eulerAngles = new Vector3(0.0f, 0.0f, 0.0f);
        modelObjectChild.transform.parent = modelObjectZ.transform;

        modelObjectY = new GameObject("Parent_GO_Y");
        modelObjectY.transform.eulerAngles = new Vector3(0.0f, 0.0f, 0.0f);
        modelObjectZ.transform.parent = modelObjectY.transform;

        modelObjectX = new GameObject("Parent_GO_X");
        modelObjectX.transform.eulerAngles = new Vector3(0.0f, 0.0f, 0.0f);
        modelObjectY.transform.parent = modelObjectX.transform;

        modelObjectParent = new GameObject("Parent_GO");
        modelObjectParent.transform.eulerAngles = new Vector3(0.0f, 0.0f, 0.0f);
        modelObjectX.transform.parent = modelObjectParent.transform;
    }

    void ClearObjects()
    {
        // Remove the paperclip gameobject
        GameObject model = GameObject.Find("Parent_GO");
        Destroy(model);
        numObjectsGenerated += 1;
    }

    string GetObjectRotationAngle()
    {
        int rotX = (int) currentObjectRot.x % 360;
        int rotY = (int) currentObjectRot.y % 360;
        int rotZ = (int) currentObjectRot.z % 360;
        string rotationAngle = string.Format("{0},{1},{2}", rotX, rotY, rotZ);
        return rotationAngle;
    }

    void RotateAlongXAxis(float stride)
    {
        modelObjectX.transform.Rotate(Vector3.right, stride, Space.World);
        currentObjectRot.x += stride;
    }

    void RotateAlongYAxis(float stride)
    {
        modelObjectX.transform.Rotate(Vector3.up, stride, Space.World);
        currentObjectRot.y += stride;
    }

    void RotateAlongZAxis(float stride)
    {
        modelObjectX.transform.Rotate(Vector3.forward, stride, Space.World);
        currentObjectRot.z += stride;
    }

    void ResetPostion()
    {
        modelObjectX.transform.position = Vector3.zero;
        modelObjectY.transform.position = Vector3.zero;
        modelObjectZ.transform.position = Vector3.zero;
    }

    // Update is called once per frame
    void Update()
    {
        // Take a screenshot if a certain number of frames has passed
        if (frameIterator % screenshotDelay == 0) {  // Number of frames has passed
            int objectIdx = numObjectsGenerated;
            
            string outputdir = string.Format("{0}/model_{1}{2}", screenshotOutputDir, objectIdx, outputDirPostfix);
            System.IO.Directory.CreateDirectory(outputdir);
            string rotationAngle = GetObjectRotationAngle();
            string outputfile = string.Format("{0}/frame_{1}_angle_{2}.{3}", outputdir, numScreenshots, rotationAngle, cameraObject.GetComponent<ScreenRecorder>().format.ToString().ToLower());

            cameraObject.GetComponent<ScreenRecorder>().takeScreenshot(outputfile);  // Take the screenshot
            numScreenshots += 1;

            if (numScreenshots >= maxScreenshots) {  // Reset the game state
                if (objectIdx == maxObjects - 1) {  // Termination condition
                    Debug.Log("All objects generated. Terminating application...");
                    Application.Quit();
                    UnityEditor.EditorApplication.isPlaying = false;  // Need to also stop from the editor in case of dev environment
                }
                ClearObjects();
                Load3DModel();
                return;
            }
        }
        
        // Update the object pose
        if (rotationAxis == "xy" || rotationAxis == "yx") {
            RotateAlongYAxis(multiAxisRotationStride);
            if (numScreenshots % multiAxisScreenshots == 0) {
                RotateAlongXAxis(multiAxisRotationStride);
            }
        }
        else if (rotationAxis == "xz" || rotationAxis == "zx") {
            RotateAlongZAxis(multiAxisRotationStride);
            if (numScreenshots % multiAxisScreenshots == 0) {
                RotateAlongXAxis(multiAxisRotationStride);
            }
        }
        else if (rotationAxis == "yz" || rotationAxis == "zy") {
            RotateAlongZAxis(multiAxisRotationStride);
            if (numScreenshots % multiAxisScreenshots == 0) {
                RotateAlongYAxis(multiAxisRotationStride);
            }
        }
        else if (rotationAxis == "xyz" || rotationAxis == "zyx" || rotationAxis == "yzx" || rotationAxis == "xzy") {
            RotateAlongZAxis(multiAxisRotationStride);
            if (numScreenshots % multiAxisScreenshots == 0) {
                RotateAlongYAxis(multiAxisRotationStride);
            }
            if (numScreenshots % (multiAxisScreenshots * multiAxisScreenshots) == 0) {
                RotateAlongXAxis(multiAxisRotationStride);
            }
        }
        else if (rotationAxis == "x") {
            RotateAlongXAxis(axisRotationStride);
        }
        else if (rotationAxis == "y") {
            RotateAlongYAxis(axisRotationStride);
        }
        else {
            Assert.IsTrue(rotationAxis == "z");
            RotateAlongZAxis(axisRotationStride);
        }

        // RotateAround modifies the position as well as the rotation -- keep the position fixed
        ResetPostion();

        frameIterator += 1;

        if (iterateOverAxis) {
            if (numScreenshots > 0 && numScreenshots % cummulativeAxisScreenshotsList[currentAxisIterator] == 0) {  // Change the rotation axis
                currentAxisIterator += 1;
                rotationAxis = axisList[currentAxisIterator];
                outputDirPostfix = "/" + rotationAxis;
                Debug.Log("Rotation axis selected to be " + rotationAxis + " axis after " + numScreenshots.ToString() + " screenshots...");
            }
        }
    }
}
