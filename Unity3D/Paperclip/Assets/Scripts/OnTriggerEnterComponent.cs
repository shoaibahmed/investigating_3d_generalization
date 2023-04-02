using System.IO;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;

public class OnTriggerEnterComponent : MonoBehaviour
{
    void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.name == "Cylinder") {
            Debug.Log("Collision detected of " + gameObject.name + " (" + gameObject.transform.position + ") with " + collision.gameObject.name + " (" + collision.gameObject.transform.position + ")");
            GameObject scriptGO = GameObject.Find("ScriptGO");
            scriptGO.GetComponent<PaperclipMultiRot>().IncrementCollisionCounter();
        }
    }

    void OnTriggerEnter(Collider collision)
    {
        if (collision.gameObject.name == "Cylinder") {
            Debug.Log("Trigger detected for collision of " + gameObject.name + " (" + gameObject.transform.position + ") with " + collision.gameObject.name + " (" + collision.gameObject.transform.position + ")");
            GameObject scriptGO = GameObject.Find("ScriptGO");
            scriptGO.GetComponent<PaperclipMultiRot>().IncrementCollisionCounter();
        }
    }
}
