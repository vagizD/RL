using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using System.Collections;
using System;
using Random = UnityEngine.Random;
using UnityEngine.InputSystem;

public class Milestone {
    public float requiredProgress;
    public float reward;
    public bool isReached;
}

public class DoggyAgent : Agent
{
    [Header("Сервоприводы")]
    public ArticulationBody[] legs;

    [Header("Скорость работы сервоприводов")]
    public float servoSpeed;

    [Header("Тело")]
    public ArticulationBody body;
    private Vector3 defPos;
    private Quaternion defRot;
    public float strenghtMove;

    [Header("Куб (цель)")]
    public GameObject cube;

    [Header("Сенсоры")]
    public Unity.MLAgentsExamples.GroundContact[] groundContacts;

    private float distToTarget = 0f;
    private float prevDistToTarget = -1f;

    private int numMilestones = 9;
    private Milestone[] milestones;

    //private Oscillator m_Oscillator;

    public override void Initialize()
    {
        distToTarget = Vector3.Distance(body.transform.position, cube.transform.position);
        defRot = body.transform.rotation;
        defPos = body.transform.position;

        //m_Oscillator = GetComponent<Oscillator>(); ***
        //m_Oscillator.ManagedReset(); ***

        milestones = new Milestone[]
        {
            new Milestone {requiredProgress = 0.10f, reward = 0.05f, isReached = false},
            new Milestone {requiredProgress = 0.20f, reward = 0.05f, isReached = false},
            new Milestone {requiredProgress = 0.30f, reward = 0.05f, isReached = false},
            new Milestone {requiredProgress = 0.40f, reward = 0.05f, isReached = false},
            new Milestone {requiredProgress = 0.50f, reward = 0.10f, isReached = false},
            new Milestone {requiredProgress = 0.60f, reward = 0.20f, isReached = false},
            new Milestone {requiredProgress = 0.70f, reward = 0.30f, isReached = false},
            new Milestone {requiredProgress = 0.80f, reward = 0.30f, isReached = false},
            new Milestone {requiredProgress = 0.90f, reward = 1.00f, isReached = false},
        };
    }

    public void ResetDog()
    {
        Quaternion newRot = Quaternion.Euler(-90, 0, Random.Range(0f, 360f));

        body.TeleportRoot(defPos, newRot);
        //body.TeleportRoot(defPos, defRot); ***
        body.velocity = Vector3.zero;
        body.angularVelocity = Vector3.zero;

        for (int i = 0; i < 12; i++)
        {
            //MoveLeg(legs[i], Random.Range(legs[i].xDrive.lowerLimit, legs[i].xDrive.upperLimit));
            MoveLeg(legs[i], 0);
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        Debug.Log("Heuristic");
    }

    public override void OnEpisodeBegin()
    {
        ResetDog();
        //m_Oscillator.ManagedReset(); ***

        //cube.transform.position = new Vector3(5, 0.21f, Random.Range(-2f, 2f));
        cube.transform.position = new Vector3(Random.Range(-7.5f, 7.5f), 0.21f, Random.Range(-7.5f, 7.5f));
        //cube.transform.position = new Vector3(5f, 0.21f, 0); ***

        //cube.transform.position = new Vector3(8f, 0.26f, 0f);

        // reinitialize distance to cube
        distToTarget = Vector3.Distance(body.transform.position, cube.transform.position);

        // reinitialize milestones progress
        for (int i = 0; i < numMilestones; ++i) {
            milestones[i].isReached = false;
        }

        // reinitialize previous target distance
        prevDistToTarget = distToTarget;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
//        sensor.AddObservation(body.transform.position);
        sensor.AddObservation(body.velocity);
        sensor.AddObservation(body.angularVelocity);
        sensor.AddObservation(body.transform.right);

        // Позиция куба
//        sensor.AddObservation(cube.transform.position);

        // Относительное положение куба
        Vector3 relativePosition = cube.transform.position - body.transform.position;
        sensor.AddObservation(relativePosition);

        // Угловая позиция куба
        Vector3 toCube = (cube.transform.position - body.transform.position).normalized;
        float angleToCube = Vector3.SignedAngle(body.transform.right, toCube, Vector3.up);
        sensor.AddObservation(angleToCube);

        // Расстояние до куба
        float distanceToCube = Vector3.Distance(body.transform.position, cube.transform.position);
        sensor.AddObservation(distanceToCube);
        foreach (var leg in legs)
        {
            sensor.AddObservation(leg.xDrive.target);
            sensor.AddObservation(leg.velocity);
            sensor.AddObservation(leg.angularVelocity);
        }

        foreach(var groundContact in groundContacts)
        {
            sensor.AddObservation(groundContact.touchingGround);
        }

        // Направление к кубу
        Vector3 toTargetWorld = (cube.transform.position - body.transform.position).normalized;
        Vector3 toTargetLocal = body.transform.InverseTransformDirection(toTargetWorld);

        sensor.AddObservation(toTargetLocal);
    }

    public override void OnActionReceived(ActionBuffers vectorAction)
    {
        var actions = vectorAction.ContinuousActions;
        for (int i = 0; i < 12; i++)
        {
            float angle = Mathf.Lerp(legs[i].xDrive.lowerLimit, legs[i].xDrive.upperLimit, (actions[i] + 1) * 0.5f);
            MoveLeg(legs[i], angle);
        }

        //m_Oscillator.ManagedUpdate(); ***

        float curDistToTarget = Vector3.Distance(body.transform.position, cube.transform.position);

        // REWARDS
        // timestep penalty (longer episode -> more total penalty)
        AddReward(-0.001f);

        // alignment rewards
        Vector3 forward = -body.transform.right;  // direction in which agent looks
        Vector3 toTarget = (cube.transform.position - body.transform.position).normalized;
        float align = Vector3.Dot(forward, toTarget);
        AddReward(0.0005f * align);  // forces to look directly to cube, enabling easier movement

//        float angle = Vector3.Angle(forward, toTarget);  // not signed, [0...180]
//        float anglePenalty = angle / 180f;  // [0...1]
//        AddReward(-0.001f * anglePenalty);  // forces to look to cube, enabling easier movement

        // movement direction rewards
        // forces the agent to walk towards the cube only when it is properly aligned
        // so, forces alignment first, and only then the movement
        float gate = Mathf.Clamp01((align - 0.2f) / (1.0f - 0.2f));  // 0 if align < 0.2
        float delta = prevDistToTarget - curDistToTarget;
        prevDistToTarget = curDistToTarget;
        delta = Mathf.Clamp(delta, -0.02f, 0.02f);
        AddReward(gate * delta);

        // milestone rewards
        float effectiveDist = Mathf.Max(distToTarget, 2.5f);         // allows all milestones to be reached
        float curProgress = 1.0f - curDistToTarget / effectiveDist;  // progress to target
        for (int i = 0; i < numMilestones; ++i) {
            if (!milestones[i].isReached && curProgress >= milestones[i].requiredProgress) {
                milestones[i].isReached = true;
                AddReward(milestones[i].reward);

                Debug.Log("Milestone progress: " + milestones[i].requiredProgress + ", reward: " + milestones[i].reward);
            }
        }

        // success
        if (curDistToTarget < 2.0f) {
            AddReward(2.0f);
            Debug.Log("Success");
            EndEpisode();
        }
    }

    public void FixedUpdate()
    {
        body.AddForce((cube.transform.position - body.transform.position).normalized * strenghtMove);
        for (int i = 0; i < 12; i++)
        {
            legs[i].AddForce((cube.transform.position - body.transform.position).normalized * strenghtMove / 20f);
        }

        RaycastHit hit;
        if (Physics.Raycast(body.transform.position, body.transform.right, out hit))
        {
            if (hit.collider.gameObject == cube)
            {
                body.AddForce(2f * strenghtMove * (cube.transform.position - body.transform.position).normalized);
                for (int i = 0; i < 12; i++)
                {
                    legs[i].AddForce((cube.transform.position - body.transform.position).normalized * strenghtMove / 10f);
                }
            }
        }
        Debug.DrawRay(body.transform.position, body.transform.right, Color.white);
    }

    void MoveLeg(ArticulationBody leg, float targetAngle)
    {
        leg.GetComponent<Leg>().MoveLeg(targetAngle, servoSpeed);
    }
}
