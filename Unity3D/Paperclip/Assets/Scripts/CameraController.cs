using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraController : MonoBehaviour
{
    // private float speedMod = 3.0f;  //a speed modifier

    // Start is called before the first frame update
    void Start()
    {
        transform.position = new Vector3(0.0f, 0.0f, -15.0f);
        transform.LookAt(Vector3.zero);
    }

    // Update is called once per frame
    void Update()
    {
        // transform.RotateAround(Vector3.zero, new Vector3(0.0f,1.0f,0.0f), 20 * Time.deltaTime * speedMod);
    }
}
