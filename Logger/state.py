StateSnapshot
(values={'messages': [
    HumanMessage(content='what is the weather in sf', additional_kwargs={}, response_metadata={}, id='2829d792-fe95-48f8-b20e-0a92694e50f4'), 
    AIMessage(content='', additional_kwargs={
        'tool_calls': [
                {'id': 'call_Juc9cxrEMVWluts7oVvOT7OM', 
            'function': {
                'arguments': '{
                            "city": "San Francisco"
                        }', 'name': 'get_weather'
                    }, 
                'type': 'function'
                }
            ], 
        'refusal': None
        }, 
        response_metadata={
            'token_usage': {'completion_tokens': 15, 
                'prompt_tokens': 56, 
                'total_tokens': 71, 
                'completion_tokens_details': {'accepted_prediction_tokens': 0, 
                    'audio_tokens': 0, 
                    'reasoning_tokens': 0, 
                    'rejected_prediction_tokens': 0
                }, 
                'prompt_tokens_details': {'audio_tokens': 0, 
                    'cached_tokens': 0
                }
            },
                'model_name': 'gpt-4.1-2025-04-14', 
                'system_fingerprint': 'fp_51e1070cf2',
                'id': 'chatcmpl-C2uRzW3DrxE8PAgPrpkYWuoBt4cT8',
                 'service_tier': 'default', 
                 'finish_reason': 'tool_calls', 
                 'logprobs': None
        }, 
        id='run--8ba5cad8-997c-4b2e-b746-4b27153ab6ad-0', 
        tool_calls=[
            {'name': 'get_weather', 
            'args': {'city': 'San Francisco'
                }, 
            'id': 'call_Juc9cxrEMVWluts7oVvOT7OM', 
            'type': 'tool_call'
            }
        ],
         usage_metadata={
            'input_tokens': 56, 
            'output_tokens': 15, 
            'total_tokens': 71, 
            'input_token_details': {'audio': 0, 'cache_read': 0
            },
             'output_token_details': {'audio': 0, 'reasoning': 0
            }
        }
        )
    ]
}, 
next=('tools',), 
config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f075b8c-4316-6c34-8001-b9c4ea7b3dc0'
    }
}, 
metadata={'source': 'loop', 'step': 1, 'parents': {}
},
created_at='2025-08-10T07: 08: 07.449066+00: 00',
parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f075b8c-3c7b-6f8a-8000-f9b1b72b623d'
    }
}, 
tasks=(
    PregelTask(id='9e2641db-3b21-eac0-14cc-c94d60e0b3d6',
     name='tools',
      path=('__pregel_push',
0, False),
       error=None, interrupts=(), 
       state=None,
        result={
            'messages': 
            [
                ToolMessage(content="It's always sunny in San Francisco!", 
                name='get_weather', 
                id='35f162ad-6f81-415a-bb6a-48e7cd1cb5bf', 
                tool_call_id='call_Juc9cxrEMVWluts7oVvOT7OM')
    ]
}),), 
interrupts=())