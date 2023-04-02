using System.IO;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;

public class PaperclipMultiRot : MonoBehaviour
{
    public GameObject cameraObject;
    private GameObject paperclipObjectX;  // Parent object used for rotation around x-axis
    private GameObject paperclipObjectY;  // Parent object used for rotation around y-axis
    private GameObject paperclipObjectZ;  // Parent object used for rotation around z-axis
    private GameObject paperclipObjectParent;  // Contains the all the other models i.e. the parent object
    private GameObject paperclipObject;
    private float maxSizeLimit;
    private float screenshotDelay;
    private int maxScreenshots;
    private int numScreenshots;
    private int frameIterator;
    private int jitteredPaperClipVariants;
    private int numPaperclipsGenerated = 0;
    private List<Vector3> sourceVertexList;
    private float jitterSigma;
    private float vertexValRange;
    private string screenshotOutputDir;
    private int maxObjects;
    private int seedVal;
    private bool reloadSeedVal;
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
    private Vector3 currentObjectRot;

    public int collisionCounter = 0;

    private List<GameObject> paperclipJoints;
    private List<string> coordList;
    private string loggingType;

    void Start()
    {
        // Disable extention logging
        Debug.unityLogger.filterLogType = LogType.Exception;
        
        // Set global variables
        reloadSeedVal = false;  // Will assume seed value already exists, and try to reload those values
        loggingType = "coords_images";

        rotationAxis = "y";
        axisList = new List<string>();
        axisList.Add("x");
        axisList.Add("y");
        axisList.Add("z");
        axisList.Add("xy");
        axisList.Add("xz");
        axisList.Add("yz");
        // axisList.Add("xyz");
        // multiAxisRotationStride = 36.0f;
        axisRotationStride = 1.0f;
        multiAxisRotationStride = 10.0f;
        maxScreenshots = 360;
        multiAxisScreenshots = maxScreenshots / (int) multiAxisRotationStride;
        axisScreenshots = maxScreenshots / (int) axisRotationStride;
        iterateOverAxis = true;
        outputDirPostfix = "";
        Debug.Log("Axis screenshots: " + axisScreenshots.ToString() + " | Multi-axis screenshots: " + multiAxisScreenshots.ToString());

        List<string> permittedStrings = new List<string> { "coords", "images", "coords_images" };
        Assert.IsTrue(permittedStrings.Contains(loggingType));

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

        maxObjects = 10000;  // Max number of paperclip objects to generate -- doesn't count the jittered versions
        maxSizeLimit = 5.0f;
        jitteredPaperClipVariants = 1;  // Generate 3 different jittered version of the same object to obtain hard negatives
        numPaperclipsGenerated = 0;
        vertexValRange = 5.0f;
        jitterSigma = vertexValRange / 5;  // Perturb by generating a random number between [-sigma, sigma]        

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
        
        GeneratePaperclipObject();
    }

    // Start is called before the first frame update
    GameObject GetCircularCylinder(Vector3 pointA, Vector3 pointB, Transform parentTransform, float radius) {
        float pointDist = Vector3.Distance(pointA, pointB);
        float yScale = pointDist / 2.0f;  // Height of a cylinder is symmetric on both sides, so the default scale is 2

        // Create the primary game object
        GameObject finalObject = new GameObject();
        finalObject.transform.parent = parentTransform;
        finalObject.name = "CircularCylinder";
        
        // Create the primary cylinder
        GameObject cylinder = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        cylinder.transform.parent = finalObject.transform;
        cylinder.transform.localScale = new Vector3(radius, yScale, radius);
        Vector3 pos = Vector3.Lerp(pointA, pointB, 0.5f);
        cylinder.transform.position = pos;
        cylinder.transform.up = pointB - pointA;

        CapsuleCollider capsuleCollider = cylinder.GetComponent<CapsuleCollider>();
        capsuleCollider.isTrigger = true;  // Make the collider a trigger
        capsuleCollider.height = yScale - radius;  // Reduce size of the collider to accommodate joints

        Rigidbody rb = cylinder.AddComponent<Rigidbody>();
        rb.isKinematic = true;
        rb.useGravity = false;
        cylinder.AddComponent<OnTriggerEnterComponent>();  // Add the collision function
        
        // Place the sphere at the end-points of the cylinder
        GameObject firstSphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        firstSphere.name = "Sphere1";
        firstSphere.transform.parent = finalObject.transform;
        firstSphere.transform.position = new Vector3(pointA.x, pointA.y, pointA.z);
        firstSphere.transform.localScale = new Vector3(radius, radius, radius);

        GameObject secondSphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        secondSphere.name = "Sphere2";
        secondSphere.transform.parent = finalObject.transform;
        secondSphere.transform.position = new Vector3(pointB.x, pointB.y, pointB.z);
        secondSphere.transform.localScale = new Vector3(radius, radius, radius);

        return finalObject;
    }

    public void IncrementCollisionCounter()
    {
        collisionCounter += 1;
    }

    void DeletePaperclipObject()
    {
        // Remove the paperclip gameobject
        GameObject paperclip = GameObject.Find("Parent_GO");
        Destroy(paperclip);
    }

    void ClearObjects()
    {
        DeletePaperclipObject();
        numPaperclipsGenerated += 1;
    }

    List<Vector3> GenerateRandomVertices() {
        List<Vector3> vertexList = new List<Vector3>();
        // int numVertices = Random.Range(6, 10);
        int numVertices = 8;  // Fixed number of vertices
        Debug.Log("# vertices: " + numVertices.ToString());

        // Constrain the movement for new vertices to ensure no hard edges are introduced via restricting the angle
        Vector3 vertex;
        for(int i = 0; i < numVertices; i++) {
            bool satisfied = false;
            while (!satisfied)
            {
                vertex = new Vector3(Random.Range(-vertexValRange, vertexValRange), Random.Range(-vertexValRange, vertexValRange), Random.Range(-vertexValRange, vertexValRange));
                if (i > 0) {
                    Vector3 offsetVertex = vertex + vertexList[i-1];  // Consider this to be an offset
                    if (i > 1) {
                        // Calculate the slope between lines
                        Vector3 firstLine = vertexList[i-1] - vertexList[i-2];
                        Vector3 secondLine = offsetVertex - vertexList[i-1];
                        // Cross the vectors to get a perpendicular vector, then normalize it.
                        float dotPr = Vector3.Dot(firstLine, secondLine);
                        float angleRad = Mathf.Acos(dotPr / (firstLine.magnitude * secondLine.magnitude));
                        float angleDeg = angleRad * Mathf.Rad2Deg;
                        Debug.Log("Dot prod: " + dotPr.ToString() + " / Vec1 norm: " + firstLine.magnitude.ToString() + " / Vec2 norm: " + secondLine.magnitude.ToString() + " / Angle between vectors: " + angleDeg.ToString());
                        if (angleDeg > 150) {
                            continue;  // Discard hard angles
                        }
                    }
                    vertex = offsetVertex;
                }
                satisfied = true;
                vertexList.Add(vertex);
            }
        }

        return vertexList;
    }

    List<Vector3> PerturbVertices(List<Vector3> vertexList) {
        List<Vector3> newVertexList = new List<Vector3>();

        for(int i = 0; i < vertexList.Count; i++) {
            Vector3 jitter = new Vector3(Random.Range(-jitterSigma, jitterSigma), Random.Range(-jitterSigma, jitterSigma), Random.Range(-jitterSigma, jitterSigma));
            Vector3 vertex = jitter + vertexList[i];
            newVertexList.Add(vertex);
        }
        return newVertexList;
    }

    void NormalizeVertices(List<Vector3> vertexList) {
        int numVertices = vertexList.Count;
        Vector3 meanVector = Vector3.zero;
        for(int i = 0; i < numVertices; i++) {
            meanVector = meanVector + vertexList[i];
        }
        meanVector = meanVector / numVertices;
        Debug.Log("Mean vector: " + meanVector.ToString());

        // Subtract the mean position from the set of vertices to align them at the center of the screen
        float maxVal = 0.0f;
        for(int i = 0; i < numVertices; i++) {
            vertexList[i] = vertexList[i] - meanVector;  // Remove the mean

            maxVal = Mathf.Max(Mathf.Abs(vertexList[i].x), maxVal);
            maxVal = Mathf.Max(Mathf.Abs(vertexList[i].y), maxVal);
            maxVal = Mathf.Max(Mathf.Abs(vertexList[i].z), maxVal);
        }
        Debug.Log("Max vertex val: " + maxVal.ToString());

        for(int i = 0; i < numVertices; i++) {
            vertexList[i] = vertexList[i] / maxVal;  // Scale each axis to unit length
            vertexList[i] = vertexList[i] * maxSizeLimit;  // Scale back to max length
        }
    }

    void GeneratePaperclipObject()
    {
        // Reset global vars
        frameIterator = 0;
        numScreenshots = 0;
        collisionCounter = 0;
        currentObjectRot = new Vector3(0.0f, 0.0f, 0.0f);

        paperclipJoints = new List<GameObject>();
        coordList = new List<string>();

        currentAxisIterator = 0;
        rotationAxis = axisList[currentAxisIterator];
        outputDirPostfix = "/" + rotationAxis;
        Debug.Log("Rotation axis selected to be " + rotationAxis + " axis after " + numScreenshots.ToString() + " screenshots...");

        List<Vector3> vertexList;
        if (numPaperclipsGenerated % jitteredPaperClipVariants == 0) {
            SetSeedVal();  // Set the seed value for reproducability
            sourceVertexList = GenerateRandomVertices();  // Generate new paperclip object
            vertexList = sourceVertexList;
        } else {
            vertexList = PerturbVertices(sourceVertexList);  // Generate a jittered version of the original paperclip
        }
        
        Debug.Log("Original vertices: " + vertexList[0].ToString());
        NormalizeVertices(vertexList);
        Debug.Log("Updated vertices: " + vertexList[0].ToString());

        // Create the primary game object
        paperclipObject = new GameObject();
        paperclipObject.name = "Paperclip";

        Vector3 startPos, endPos;
        for(int i = 1; i < vertexList.Count; i++) {
            startPos = vertexList[i-1];
            endPos = vertexList[i];
            GameObject cylinder = GetCircularCylinder(startPos, endPos, paperclipObject.transform, 0.5f);

            // Get the sphere locations from here
            foreach(Transform childTransform in cylinder.transform)
            {
                GameObject childGameObject = childTransform.gameObject; // Convert the Transform to a GameObject

                if (childGameObject.name == "Sphere1") {
                    if (i == 1) {  // Add the first joint only for the first time
                        paperclipJoints.Add(childGameObject);
                    }
                } else if (childGameObject.name == "Sphere2") {
                    paperclipJoints.Add(childGameObject);
                }
            }
        }
        Debug.Log("# joints tracked: " + paperclipJoints.Count.ToString());

        bool randomObjectRotation = false;
        if (randomObjectRotation) {
            // Rotate the object randomly before rotating the parent
            // paperclipObject.transform.localEulerAngles = new Vector3(Random.Range(0, 360), Random.Range(0, 360), Random.Range(0, 360));
            paperclipObject.transform.rotation = Random.rotation;
        }
        else {
            paperclipObject.transform.eulerAngles = new Vector3(0.0f, 0.0f, 0.0f);
        }

        // Assign the main game object as parent which will be used for rotation
        paperclipObjectZ = new GameObject("Parent_GO_Z");
        paperclipObjectZ.transform.eulerAngles = new Vector3(0.0f, 0.0f, 0.0f);
        paperclipObject.transform.parent = paperclipObjectZ.transform;

        paperclipObjectY = new GameObject("Parent_GO_Y");
        paperclipObjectY.transform.eulerAngles = new Vector3(0.0f, 0.0f, 0.0f);
        paperclipObjectZ.transform.parent = paperclipObjectY.transform;

        paperclipObjectX = new GameObject("Parent_GO_X");
        paperclipObjectX.transform.eulerAngles = new Vector3(0.0f, 0.0f, 0.0f);
        paperclipObjectY.transform.parent = paperclipObjectX.transform;

        paperclipObjectParent = new GameObject("Parent_GO");
        paperclipObjectParent.transform.eulerAngles = new Vector3(0.0f, 0.0f, 0.0f);
        paperclipObjectX.transform.parent = paperclipObjectParent.transform;
    }

    int LoadJson(string fileName)
    {
        using (StreamReader r = new StreamReader(fileName))
        {
            string json = r.ReadToEnd();
            Debug.Log("JSON: " + json);
            json = json.Replace("{\"seed\":", "").Replace("}", "");
            Debug.Log("Replaced JSON: " + json);
            int seed = int.Parse(json);
            return seed;
        }
    }

    void SetSeedVal()
    {
        int paperclipIdx = (int) numPaperclipsGenerated / jitteredPaperClipVariants;
        string outputdir = string.Format("{0}/paperclip_{1}/", screenshotOutputDir, paperclipIdx);
        string outputfile = outputdir + "seed.json";

        if (reloadSeedVal) {
            seedVal = LoadJson(outputfile);
        }
        else {
            // Seed the process for reproducability
            seedVal = (int) System.DateTime.Now.Ticks;  // Random.Range(0, 100000);
        }

        Random.InitState(seedVal);
        Debug.Log("Selected seed value: " + seedVal.ToString());

        // Write seed value to a file
        System.IO.Directory.CreateDirectory(outputdir);

        try
        {
            StreamWriter sw = new StreamWriter(outputfile);
            sw.WriteLine("{\"seed\":" + seedVal.ToString() + "}");
            sw.Close();
        }
        catch(System.Exception e)
        {
            Debug.Log("Exception: " + e.Message);
        }
        finally
        {
            Debug.Log("Executing finally block for seed writer.");
        }
    }

    int GetXRot()
    {
        int rotX = (int) currentObjectRot.x % 360;
        return rotX;
    }

    int GetYRot()
    {
        int rotY = (int) currentObjectRot.y % 360;
        return rotY;
    }

    int GetZRot()
    {
        int rotZ = (int) currentObjectRot.z % 360;
        return rotZ;
    }

    string GetObjectRotationAngle()
    {
        int rotX = GetXRot();
        int rotY = GetYRot();
        int rotZ = GetZRot();
        string rotationAngle = string.Format("{0},{1},{2}", rotX, rotY, rotZ);
        return rotationAngle;
    }

    void RotateAlongXAxis(float stride)
    {
        paperclipObjectX.transform.Rotate(Vector3.right, stride, Space.World);
        currentObjectRot.x += stride;
    }

    void RotateAlongYAxis(float stride)
    {
        paperclipObjectX.transform.Rotate(Vector3.up, stride, Space.World);
        currentObjectRot.y += stride;
    }

    void RotateAlongZAxis(float stride)
    {
        paperclipObjectX.transform.Rotate(Vector3.forward, stride, Space.World);
        currentObjectRot.z += stride;
    }

    void ResetPostion()
    {
        paperclipObjectX.transform.position = Vector3.zero;
        paperclipObjectY.transform.position = Vector3.zero;
        paperclipObjectZ.transform.position = Vector3.zero;
    }

    string ConvertToJSON(Vector3 input)
    {
        return input.ToString().Replace('(', '[').Replace(')', ']');
    }

    // Update is called once per frame
    void Update()
    {
        if (collisionCounter > 0) {
            int paperclipIdx = (int) numPaperclipsGenerated / jitteredPaperClipVariants;
            int hardNegativeIdx = numPaperclipsGenerated % jitteredPaperClipVariants;
            Debug.Log("[Warning] Paperclip collisions is non-zero (" + collisionCounter.ToString() + ") for paperclip ID " + paperclipIdx.ToString() + " (Hard negative:" + hardNegativeIdx.ToString() + "). Regenerating Paperclip!");
            DeletePaperclipObject();
            GeneratePaperclipObject();
            return;  // Do nothing on this frame. Restart with the test at the next iteration.
        }

        // Take a screenshot if a certain number of frames has passed
        if (frameIterator % screenshotDelay == 0) {  // Number of frames has passed
            int paperclipIdx = (int) numPaperclipsGenerated / jitteredPaperClipVariants;
            int hardNegativeIdx = numPaperclipsGenerated % jitteredPaperClipVariants;

            if (loggingType == "images" || loggingType == "coords_images") {
                string outputdir = string.Format("{0}/paperclip_{1}/hard_negative_{2}{3}", screenshotOutputDir, paperclipIdx, hardNegativeIdx, outputDirPostfix);
                System.IO.Directory.CreateDirectory(outputdir);
                string rotationAngle = GetObjectRotationAngle();
                string outputfile = string.Format("{0}/frame_{1}_angle_{2}.{3}", outputdir, numScreenshots, rotationAngle, cameraObject.GetComponent<ScreenRecorder>().format.ToString().ToLower());

                cameraObject.GetComponent<ScreenRecorder>().takeScreenshot(outputfile);  // Take the screenshot
            }
            
            if (loggingType == "coords" || loggingType == "coords_images") {
                if (numScreenshots == 0)  // Just reference img
                {
                    string outputfile = string.Format("{0}/paperclip_{1}/reference_img.{2}", screenshotOutputDir, paperclipIdx, cameraObject.GetComponent<ScreenRecorder>().format.ToString().ToLower());
                    cameraObject.GetComponent<ScreenRecorder>().takeScreenshot(outputfile);  // Take the screenshot
                }

                Camera camera = cameraObject.GetComponent<Camera>();
                string coords = "{\"rot\": {\"x\": " + GetXRot().ToString() + ", \"y\": " + GetYRot().ToString() + ", \"z\": " + GetZRot().ToString() + "}";
                coords += ", \"rot_axis\": \"" + rotationAxis + "\"";
                for (int i = 0; i < paperclipJoints.Count; i++) {
                    // Get the position of the GameObject in world space
                    Vector3 worldPos = paperclipJoints[i].transform.position;
                    
                    // Convert the world space position to viewport space
                    Vector3 viewportPos = camera.WorldToViewportPoint(worldPos);

                    // Write image coordinates in the form 0-1
                    float imageX = viewportPos.x;
                    float imageY = (1 - viewportPos.y);

                    string worldPosString = "\"world\": " + ConvertToJSON(worldPos);
                    string viewportPosString = "\"viewport\": " + ConvertToJSON(viewportPos);
                    string imagePosString = "\"image\": [" + imageX.ToString() + ", " + imageY.ToString() + "]";

                    coords = coords + ", \"j" + i + "\"" + ": {" + worldPosString + ", " + viewportPosString + ", " + imagePosString + "}";
                }
                coords += "}";
                Debug.Log("Tracker / " + coords);
                coordList.Add(coords);
            }

            numScreenshots += 1;

            if (numScreenshots >= maxScreenshots) {  // Reset the game state
                if ((paperclipIdx == (maxObjects - 1)) && (hardNegativeIdx == (jitteredPaperClipVariants - 1))) {  // Termination condition
                    Debug.Log("All objects generated. Terminating application...");
                    Application.Quit();
                    UnityEditor.EditorApplication.isPlaying = false;  // Need to also stop from the editor in case of dev environment
                }

                // Write the final coords to file
                string outputdir = string.Format("{0}/paperclip_{1}/", screenshotOutputDir, paperclipIdx);
                string outputfile = outputdir + "coords.jsonl";

                try
                {
                    StreamWriter sw = new StreamWriter(outputfile);
                    for (int i = 0; i < coordList.Count; i++) {
                        sw.WriteLine(coordList[i]);
                    }
                    sw.Close();
                }
                catch(System.Exception e)
                {
                    Debug.Log("Exception: " + e.Message);
                }
                finally
                {
                    Debug.Log("Executing finally block for coords writer.");
                }
                
                ClearObjects();
                GeneratePaperclipObject();
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
